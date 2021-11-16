
from backdoor import BackDoor
def ble_inj(args):
    print(args)
    BackDoor.blended_injection(args.input_csv,args.output_csv,args.target_label,args.key_pattern,args.poison_num,args.alpha)
    pass