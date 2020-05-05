import re
import sys


def prepare(num_dialogs=20000):
    with open("../data/xiaohuangji50w_nofenci.conv","r",encoding="utf-8") as f:
        reg = re.compile("E\nM (.*?)\nM (.*?)\n")
        match_dialogs = re.findall(reg, f.read())
        # if num_dialogs >= len(match_dialogs):
        #     dialogs = match_dialogs
        # else:
        #     dialogs = match_dialogs[:num_dialogs]

        questions = []
        answers = []
        for que, ans in match_dialogs:
            questions.append(que)
            answers.append(ans)
        save(questions, "dialog/Q")
        save(answers, "dialog/A")


def save(dialogs, file):
    with open(file, "w",encoding="utf-8") as fopen:
        fopen.write("\n".join(dialogs))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        num_dialogs = int(sys.argv[1])
        prepare(num_dialogs)
    else:
        prepare()