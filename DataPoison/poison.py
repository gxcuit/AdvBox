import argparse

import cmd.generate

def hello(args):
    print('-----')
    print(args)

def main():
    parser = argparse.ArgumentParser(prog='poison', description='poison')
    subparsers = parser.add_subparsers()
    generate_parser = subparsers.add_parser('generate')
    subparser2 = generate_parser.add_subparsers()

    generate_bi_parser = subparser2.add_parser('blended-injection')

    generate_bi_parser.add_argument('--input-csv', '-i')
    generate_bi_parser.add_argument('--output-csv', '-o')
    generate_bi_parser.add_argument('--target-label', '-t',type=int)
    generate_bi_parser.add_argument('--key-pattern', '-k')
    generate_bi_parser.add_argument('--poison-num', '-n',type=int)
    generate_bi_parser.add_argument('--alpha', '-a',default=0.2,type=float)
    generate_parser.set_defaults(func=cmd.generate.ble_inj)



    args = parser.parse_args()
    args.func(args)
    #print(args.generate_poison_train)


if __name__ == '__main__':
    main()