from ml_utils.config_parser import ConfigParser
from ml_utils.console_args import args

from model.sem_eval_model import SemEvalModel
from src.tasks.infer import infer_from_trained

config = ConfigParser(
    args.config_file, console_args=dict(args._get_kwargs())
).parse()


if __name__ == "__main__":
    fine_tuner = SemEvalModel(config)
    net = fine_tuner.train()

    inferer = infer_from_trained(args, detect_entities=True)
    test = "The surprise [E1]visit[/E1] caused a [E2]frenzy[/E2] on the already chaotic trading floor."
    inferer.infer_sentence(test, detect_entities=False)
    test2 = "After eating the chicken, he developed a sore throat the next morning."
    inferer.infer_sentence(test2, detect_entities=True)

    while True:
        sent = input("Type input sentence ('quit' or 'exit' to terminate):\n")
        if sent.lower() in ["quit", "exit"]:
            break
        inferer.infer_sentence(sent, detect_entities=False)
