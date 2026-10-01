from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--remove_query', action='store_true')
parser.add_argument(
    '--use_input_rank_score',
    action='store_true',
    help=(
        'Write a strictly decreasing score derived from the input order. '
        'Use when the input order is intentional and equal fp16 logits must '
        'not be re-ordered by a TREC evaluator.'
    ),
)
args = parser.parse_args()

with open(args.input) as f_in, open(args.output, 'w') as f_out:
    cur_qid = None
    rank = 0
    for line in f_in:
        qid, docid, score = line.split()
        if cur_qid != qid:
            cur_qid = qid
            rank = 0
        if args.remove_query and qid == docid:
            continue
        rank += 1      
        output_score = -rank if args.use_input_rank_score else score
        f_out.write(f'{qid} Q0 {docid} {rank} {output_score} dense\n')
