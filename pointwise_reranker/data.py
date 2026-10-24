# data.py
import os
import random
import logging
from dataclasses import dataclass
from typing import List, Tuple
import torch
import datasets
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from arguments import DataArguments
from eval_text import format_eval_document

logger = logging.getLogger(__name__)

def normalize_train_example(example):
    """Adapt MVP's one-based positive index; preserve legacy passage pools."""
    if 'positive_passages' in example and 'negative_passages' in example:
        return example
    if 'ret' not in example or 'pos_idx' not in example:
        raise ValueError("Expected positive_passages/negative_passages or MVP ret/pos_idx.")
    passages = example['ret']
    index = example['pos_idx']
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"MVP pos_idx must be a one-based integer, got {index!r}")
    if not isinstance(passages, list) or len(passages) < 2 or not 1 <= index <= len(passages):
        raise ValueError("MVP needs at least two passages and pos_idx in [1, len(ret)].")
    if not isinstance(example.get('query'), str) or not example['query'].strip():
        raise ValueError("MVP query must be nonempty text.")
    if any(not isinstance(p, dict) or not isinstance(p.get('text'), str)
           or not p['text'].strip() for p in passages):
        raise ValueError("Every MVP passage must have nonempty text.")
    return {
        'query': example['query'],
        'positive_passages': [passages[index - 1]],
        'negative_passages': passages[:index - 1] + passages[index:],
    }

class HFRerankerTrainDataset:
    def __init__(self, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer,
                 data_args: DataArguments, cache_dir: str):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}

        self.dataset = load_dataset(data_args.dataset_name,
                                    data_args.dataset_language,
                                    data_files=data_files, use_auth_token=None)[data_args.dataset_split]#.select(range(1))
        
        self.preprocessor = RerankerTrainPreProcessor(tokenizer_student, tokenizer_teacher,
                                                      data_args.q_max_len, data_args.p_max_len)
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        #查询文本的最大token长度
        self.q_max_len = data_args.q_max_len
        #文档/段落的最大token长度
        self.p_max_len = data_args.p_max_len
        #数据处理进程数
        self.proc_num = data_args.dataset_proc_num
        #负样本数量
        self.neg_num = data_args.train_n_passages - 1
        self.separator = getattr(self.tokenizer_student,  # object
                                 data_args.passage_field_separator,# attribute_name
                                 data_args.passage_field_separator) # default_value

    def process(self, shard_num=1, shard_idx=0):
        #将数据集分成 shard_num 个部分，并返回第 shard_idx 个部分。
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(#对每份数据进行tokenization:将自然语言文本转换为模型能够理解的数字表示
                self.preprocessor, # 预处理函数
                batched=False, # 逐条处理（非批处理）
                num_proc=self.proc_num, # 并行进程数
                remove_columns=self.dataset.column_names,# 删除原始列
                desc="Tokenizing train dataset",# 进度条描述
            )
        return self.dataset

class RerankerTrainPreProcessor:
    def __init__(self, tokenizer_student, tokenizer_teacher, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        example = normalize_train_example(example)
        query_text = 'query: ' + example['query']#在原始查询前添加 'query: ' 前缀
        # Tokenize query
        #经过学生模型的查询token IDs
        q_stu = self.tokenizer_student.encode(query_text, # 要编码的文本
                                              add_special_tokens=False,# 不添加[CLS]/[SEP]等特殊token
                                              max_length=self.query_max_length, # 最大长度限制
                                              truncation=True) # 超出部分截断
        # 教师模型的查询token IDs
        q_tch = self.tokenizer_teacher.encode(query_text, add_special_tokens=False,
                                              max_length=self.query_max_length, truncation=True)
# 存放学生tokenized的正文档pos_stu
        pos_stu, pos_tch = [], []
        neg_stu, neg_tch = [], []
        # 遍历正文档列表
        for pos in example['positive_passages']:

            text = (pos.get('title', '') + self.separator + pos['text']) if 'text' in pos else pos.get('title', '')
            doc_text = 'document: ' + text
            pos_stu.append(self.tokenizer_student.encode(doc_text, # 文档文本
                                                         add_special_tokens=False, # 不添加特殊token
                                                         max_length=self.text_max_length - 3,  # 最大长度（预留3个位置）
                                                         truncation=True))    # 超长截断
            pos_tch.append(self.tokenizer_teacher.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))

        for neg in example['negative_passages']:
            text = (neg.get('title', '') + self.separator + neg['text']) if 'text' in neg else neg.get('title', '')
            doc_text = 'document: ' + text
            neg_stu.append(self.tokenizer_student.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))
            neg_tch.append(self.tokenizer_teacher.encode(doc_text, add_special_tokens=False,
                                                         max_length=self.text_max_length - 3, truncation=True))

        return {
            'query_student': q_stu, 'positives_student': pos_stu, 'negatives_student': neg_stu,
            'query_teacher': q_tch, 'positives_teacher': pos_tch, 'negatives_teacher': neg_tch
        }

class RerankerTrainDataset(Dataset):
    def __init__(self, data_args: DataArguments, dataset: datasets.Dataset,
                 tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer):
        self.train_data = dataset
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher# 文本 → 数字
        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(
    self,
    query_encoding: List[int],
    text_encoding: List[int],
    tokenizer,
):
        """
        保持 DeAR 原始输入格式，只额外生成 Q/D/S 区域标签。

        region_ids:
            0  = Query
            1  = Document
            2  = Score Position
            -1 = tokenizer 添加的其他特殊 token，
                或之后由 collator 添加的 Padding
        """

        max_length = (
            self.data_args.q_max_len
            + self.data_args.p_max_len
        )

        # ==================================================
        # 1. 真实模型输入：与 DeAR 原始源代码完全一致
        # ==================================================
        item = tokenizer.prepare_for_model(
            [tokenizer.bos_token_id]#bos表示序列开始的token_id
            + list(query_encoding),

            [tokenizer.bos_token_id]
            + list(text_encoding),
            add_special_tokens=False,
#当总长度超过 max_length 时，只截断第二个序列（文档），保留第一个序列（查询）完整。
            truncation="only_second",
            max_length=max_length,
            padding=False,
            return_attention_mask=False,#不生成注意力掩码([1, 1, 1, 0, 0, 0]    # 1表示真实token，0表示padding)
            return_token_type_ids=False,#不生成token类型ID
        )
# item = {
#     'input_ids': [101, 2054, 2003, 4868, 2293, 1029, 102, 101, 1064, 1078],
#     #             ↑     ↑     ↑     ↑     ↑     ↑     ↑    ↑    ↑     ↑
#     #           [CLS] what  is   climate ?   [SEP] [CLS] climate change
#     #           查询部分（完整）     ↑分隔  文档部分（被截断，只保留了前2个token）
    # attention_mask不返回
#     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 所有位置都是有效token
#     #                   ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#     #                 全部是1，因为没有padding
    # token_type_ids不返回
#     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
#     #                   ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
#     #                 第一个句子(查询)  第二个句子(文档)
# }
        # ==================================================
        # 2. 平行生成区域位置
        # marker 不会输入模型
        # Marker 是虚拟的、不会输入模型的占位符
        # ==================================================
        query_input_length = (
            1 + len(query_encoding)
        )

        document_input_length = (
            1 + len(text_encoding)
        )

        query_marker = -10000001 # 标记"查询"区域
        document_marker = -10000002 # 标记"文档"区域

        marker_item = tokenizer.prepare_for_model(
            #创建一个列表，包含 query_input_length 个 query_marker
            [query_marker]
            * query_input_length,

            [document_marker]
            * document_input_length,
            add_special_tokens=False,
            # 必须和上面的真实输入完全一致
            truncation="only_second",
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        model_input_ids = item["input_ids"]
        marker_ids = marker_item["input_ids"]

        if len(model_input_ids) != len(marker_ids):
            raise RuntimeError(
                "Model input and marker lengths differ: "
                f"{len(model_input_ids)} "
                f"!= {len(marker_ids)}"
            )

        # ==================================================
        # 3. 生成 Q/D 区域标签
        # ==================================================
        region_ids = []

        for marker in marker_ids:
            if marker == query_marker:
                region_ids.append(0)

            elif marker == document_marker:
                region_ids.append(1)

            else:
                # prepare_for_model 自动添加的特殊 token
                region_ids.append(-1)

        if len(region_ids) == 0:
            raise ValueError(
                "Encoded sequence is empty."
            )

        # ==================================================
        # 4. Score Position(=2)不在这里标注。
        # 模型固定在padding到max_length后的最后一个位置打分
        # (见modeling.use_last_position_scoring)，因此由collator
        # 在padding之后把最后一位标为2。
        # ==================================================
        if 0 not in region_ids:
            raise ValueError(
                "Query region is empty."
            )

        if 1 not in region_ids:
            raise ValueError(
                "Document region is empty."
            )

        import os

        debug_tokenizer_name = tokenizer.name_or_path

        debug_printed = getattr(
            self,
            "_alignment_debug_printed",
            set(),
        )

        if (
            os.environ.get("DEAR_DEBUG_ALIGNMENT", "0") == "1"
            and debug_tokenizer_name not in debug_printed
        ):
            debug_printed.add(debug_tokenizer_name)
            self._alignment_debug_printed = debug_printed

            query_positions = [
                i for i, r in enumerate(region_ids)
                if r == 0
            ]

            document_positions = [
                i for i, r in enumerate(region_ids)
                if r == 1
            ]

            print("\n===== Alignment Debug =====")
            print("tokenizer_role:", debug_tokenizer_name)
            print("tokenizer_path:", tokenizer.name_or_path)
            print("input_length:", len(model_input_ids))

            print(
                "query_region:",
                query_positions[:5],
                "...",
                query_positions[-5:],
            )

            print(
                "document_region:",
                document_positions[:5],
                "...",
                document_positions[-5:],
            )

            print("score_position:", max_length - 1, "(set by collator)")
            print("indexed_tokens_and_regions:")

            for pos, token_id in enumerate(model_input_ids):
                if pos >= 140:
                    break

                region_id = region_ids[pos]

                print(
                    pos,
                    "token_id=", token_id,
                    "region_id=", region_id,
                    "token=",
                    repr(
                        tokenizer.decode(
                            [token_id],
                            skip_special_tokens=False,
                        )
                    ),
                )
            query_token_ids = [
                model_input_ids[i]
                for i in query_positions
            ]

            document_token_ids = [
                model_input_ids[i]
                for i in document_positions
            ]

            print("query_token_count:", len(query_token_ids))
            print("document_token_count:", len(document_token_ids))

            print(
                "query_decoded:",
                tokenizer.decode(
                    query_token_ids,
                    skip_special_tokens=False,
                ),
            )

            print(
                "document_decoded_head:",
                tokenizer.decode(
                    document_token_ids[:20],
                    skip_special_tokens=False,
                ),
            )

            print("===========================\n")
        return item, region_ids

    def __len__(self):
        return self.total_len

    def __getitem__(
        self,
        item,
    ) -> Tuple[
        List[BatchEncoding],
        List[BatchEncoding],
    ]:
        group = self.train_data[item]

        # Student
        qry_stu = group["query_student"]
        pos_stu = group["positives_student"]
        neg_stu = group["negatives_student"]

        # Teacher
        qry_tch = group["query_teacher"]
        pos_tch = group["positives_teacher"]
        neg_tch = group["negatives_teacher"]

        feats_stu = []
        feats_tch = []

        # Student 和 Teacher 的文档池必须对齐
        assert len(pos_stu) == len(pos_tch), (
            "Student/teacher positive pools "
            "are misaligned."
        )

        assert len(neg_stu) == len(neg_tch), (
            "Student/teacher negative pools "
            "are misaligned."
        )

        # ==================================================
        # 选择正文档
        # ==================================================
        #positive_passage_no_shuffle = True：不随机打乱，总是选择第一个正文档
        if (
            self.data_args
            .positive_passage_no_shuffle
        ):
            pos_idx = 0
        else:
            pos_idx = random.randrange(
                len(pos_stu)
            )

        p_stu = pos_stu[pos_idx]
        p_tch = pos_tch[pos_idx]
#为正文档（Positive Passage）创建学生和教师的训练样本
        feats_stu.append(
            self.create_one_example(
                qry_stu,  #   学生tokenizer query的编码
                p_stu,    # 学生tokenizer正文档的编码
                self.tokenizer_student, # 学生tokenizer
            )
        )

        feats_tch.append(
            self.create_one_example(
                qry_tch,
                p_tch,
                self.tokenizer_teacher,
            )
        )

        # ==================================================
        # 选择负文档
        # ==================================================
        neg_size = (
            self.data_args.train_n_passages
            - 1
        )

        if len(neg_stu) == 0:
            raise ValueError(
                "Training sample has no negatives."
            )

        if len(neg_stu) < neg_size:#实际有的负文档数量小于需要的负文档数量
            neg_indices = random.choices(#有放回采样，重复选择已有的负文档来凑够数量
                range(len(neg_stu)),
                k=neg_size,
            )

        elif (
            self.data_args
            .negative_passage_no_shuffle
        ):
            neg_indices = list(
                range(neg_size)
            )#从 0 到 neg_size-1 的连续整数列表。

        else:
            neg_indices = random.sample(#不放回采样
                range(len(neg_stu)),
                neg_size,
            )
        #学生负文档
        ns_stu = [
            neg_stu[i]
            for i in neg_indices
        ]
        #教师负文档
        ns_tch = [
            neg_tch[i]
            for i in neg_indices
        ]

        for student_text, teacher_text in zip(
            ns_stu,
            ns_tch,
        ):
            feats_stu.append(
                self.create_one_example(
                    qry_stu,
                    student_text,
                    self.tokenizer_student,
                )
            )

            feats_tch.append(
                self.create_one_example(
                    qry_tch,
                    teacher_text,
                    self.tokenizer_teacher,
                )
            )

        return feats_stu, feats_tch


@dataclass
class RerankerTrainCollator:
    tokenizer_student: PreTrainedTokenizer
    tokenizer_teacher: PreTrainedTokenizer
    max_q_len: int = 32
    max_p_len: int = 196

    def __call__(self, features):
        student_features, teacher_features = zip(
            *features
        )
#sum把多个列表合并成一个一维列表
        student_flat = sum(student_features, [])
        teacher_flat = sum(teacher_features, [])

        student_model_features = [
            x[0] for x in student_flat
        ]

        teacher_model_features = [
            x[0] for x in teacher_flat
        ]

        student_region_lists = [
            x[1] for x in student_flat
        ]

        teacher_region_lists = [
            x[1] for x in teacher_flat
        ]

        max_length = (
            self.max_q_len + self.max_p_len
        )

        student_collated = self.tokenizer_student.pad(
            student_model_features,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        teacher_collated = self.tokenizer_teacher.pad(
            teacher_model_features,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        def pad_region_ids(region_lists):
            #创建一个形状为 (len(region_lists), max_length) 的二维张量，所有位置都初始化为 -1
            output = torch.full(
                (len(region_lists), max_length),
                fill_value=-1,
                dtype=torch.long,
            )

            for row, region_ids in enumerate(
                region_lists
            ):
                length = min(
                    len(region_ids),
                    max_length,
                )

                output[row, :length] = torch.tensor(
                    region_ids[:length],
                    dtype=torch.long,
                )

            # 模型在最后一个位置打分，该位置即Score Position
            output[:, max_length - 1] = 2

            return output

        return {
            "student": student_collated,
            "teacher": teacher_collated,
            "student_region_ids": pad_region_ids(
                student_region_lists
            ),
            "teacher_region_ids": pad_region_ids(
                teacher_region_lists
            ),
        }
class HFRerankerInferenceDataset:
    def __init__(self, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer, 
                 data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.dataset = datasets.load_dataset(data_args.dataset_name,
                                             data_args.dataset_language,
                                             data_files=data_files, use_auth_token=None)[data_args.dataset_split]
        self.preprocessor = RerankerInferencePreProcessor(tokenizer_student, tokenizer_teacher, 
                                                          data_args.q_max_len, data_args.p_max_len)
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.q_max_len = data_args.q_max_len
        self.p_max_len = data_args.p_max_len
        self.proc_num = data_args.dataset_proc_num
        self.separator = getattr(self.tokenizer_student, data_args.passage_field_separator, data_args.passage_field_separator)

    def process(self, shard_num=1, shard_idx=0):
        self.dataset = self.dataset.shard(shard_num, shard_idx)
        if self.preprocessor is not None:
            self.dataset = self.dataset.map(
                self.preprocessor,
                batched=False,
                num_proc=self.proc_num,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on inference dataset",
            )
        return self.dataset


class RerankerInferencePreProcessor:
    def __init__(self, tokenizer_student, tokenizer_teacher, query_max_length=32, text_max_length=256, separator=' '):
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.query_max_length = query_max_length
        self.text_max_length = text_max_length
        self.separator = separator

    def __call__(self, example):
        query_text = 'query: ' + example['query'] 
        doc_text = format_eval_document(example.get('title', ''), example['text'], self.separator)
        # Tokenize query and document separately for student & teacher
        query_student = self.tokenizer_student.encode(query_text, 
                                                      add_special_tokens=False, 
                                                      max_length=self.query_max_length, 
                                                      truncation=True)
        query_teacher = self.tokenizer_teacher.encode(query_text, 
                                                      add_special_tokens=False, 
                                                      max_length=self.query_max_length, 
                                                      truncation=True)

        text_student = self.tokenizer_student.encode(doc_text, 
                                                     add_special_tokens=False, 
                                                     max_length=self.text_max_length-3, 
                                                     truncation=True)
        text_teacher = self.tokenizer_teacher.encode(doc_text, 
                                                     add_special_tokens=False, 
                                                     max_length=self.text_max_length-3, 
                                                     truncation=True)

        return {
            'query_id': example['query_id'],
            'query_student': query_student, 'text_student': text_student,
            'query_teacher': query_teacher, 'text_teacher': text_teacher,
            'text_id': example['docid']
        }


class RerankerInferenceDataset(Dataset):
    input_keys = [
        "query_id",
        "query_student",
        "text_id",
        "text_student",
        "query_teacher",
        "text_teacher",
    ]

    def __init__(
        self,
        dataset: datasets.Dataset,
        tokenizer_student: PreTrainedTokenizer,
        tokenizer_teacher: PreTrainedTokenizer,
        max_q_len=32,
        max_p_len=256,
    ):
        self.encode_data = dataset
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(
        self,
        item,
    ) -> Tuple[
        str,
        str,
        BatchEncoding,
        BatchEncoding,
    ]:
        (
            query_id,
            query_student,
            text_id,
            text_student,
            query_teacher,
            text_teacher,
        ) = (
            self.encode_data[item][f]
            for f in self.input_keys
        )

        # Student 输入：DeAR 原始实现
        encoded_pair_student = (
            self.tokenizer_student.prepare_for_model(
                [
                    self.tokenizer_student.bos_token_id
                ] + query_student,
                [
                    self.tokenizer_student.bos_token_id
                ] + text_student,
                truncation="only_second",
                add_special_tokens=False,
                padding="max_length",
                max_length=(
                    self.max_q_len
                    + self.max_p_len
                ),
                return_token_type_ids=False,
            )
        )

        # Teacher 输入：DeAR 原始实现
        encoded_pair_teacher = (
            self.tokenizer_teacher.prepare_for_model(
                [
                    self.tokenizer_teacher.bos_token_id
                ] + query_teacher,
                [
                    self.tokenizer_teacher.bos_token_id
                ] + text_teacher,
                truncation="only_second",
                padding="max_length",
                max_length=(
                    self.max_q_len
                    + self.max_p_len
                ),
                return_token_type_ids=False,
            )
        )

        return (
            query_id,
            text_id,
            encoded_pair_student,
            encoded_pair_teacher,
        )
@dataclass
class RerankerInferenceCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer_student: PreTrainedTokenizer, tokenizer_teacher: PreTrainedTokenizer, 
                 max_q_len: int = 32, max_p_len: int = 196):
        self.tokenizer_student = tokenizer_student
        self.tokenizer_teacher = tokenizer_teacher
        self.max_q_len = max_q_len
        self.max_p_len = max_p_len

    def __call__(self, features):
        query_ids = [x[0] for x in features]
        text_ids = [x[1] for x in features]
        student_features = [x[2] for x in features]
        teacher_features = [x[3] for x in features]

        # Collate student data
        student_collated = self.tokenizer_student.pad(
            student_features,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )

        # Collate teacher data
        teacher_collated = self.tokenizer_teacher.pad(
            teacher_features,
            padding='max_length',
            max_length=self.max_q_len + self.max_p_len,
            return_tensors="pt",
        )
        return query_ids, text_ids, student_collated, teacher_collated
