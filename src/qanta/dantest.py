from danguesser import DanGuesser
import dan

if __name__ == '__main__':
    # dan.train()
    dg = DanGuesser.load()
    result = dan.guess_and_buzz(dg, "This is a question about science.")
    result