from qanta.danguesser import DanGuesser
import qanta.dan

if __name__ == '__main__':
    # qanta.dan.train()
    dg = DanGuesser.load()
    result = qanta.dan.guess_and_buzz(dg, "Name the the inventor of general relativity and the photoelectric effect")
    result