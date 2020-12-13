import argparse


def getConfig():
    parser = argparse.ArgumentParser()

    parser.add_argument('--random-seed', default=111, type=int, help='random seed')

    parser.add_argument('--dataset', default='cub_bird', help='name of the dataset (cub_bird or stanford_dog or aircraft or vegfru)')

    parser.add_argument('--code-length', default=16, type=int, help='length of hash codes')

    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')

    parser.add_argument('--batch-size', default=128, type=int, help='batch size')

    parser.add_argument('--epoch', default=90, type=int, help='total training epochs')

    parser.add_argument('--gpu-ids', default='6,7', help='gpu id list(eg: 0,1,2...)')

    parser.add_argument('--model-name', default='resnet18', type=str, help='model name (resnet18 or alexnet)')

    parser.add_argument('--class-mask', default=0.7, type=float, help='class mask rate')

    args = parser.parse_args()

    return args
