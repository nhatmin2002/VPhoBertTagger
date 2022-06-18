from vphoberttagger import LOGGER, Trainer, Predictor

import sys


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process...")
        Trainer.train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process...")
        Trainer.test()
    elif sys.argv[1] == 'predict':
        LOGGER.info("Start PREDICT process...")
        Predictor.tagging()
    else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')