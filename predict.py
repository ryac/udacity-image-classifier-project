# imports..
from predict_utils import load_model
from predict_utils import process_image
import torch
import argparse
import json

device = None


def predict():

    device = args.device

    # label mapping..
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    image_input = process_image(args.opts[0])
    image_input = image_input.type(torch.cuda.FloatTensor)
    image_input.unsqueeze_(0)
    image_input = image_input.to(device)

    model = load_model(args.opts[1])
    model.to(device)

    index_to_class = {value: key for key, value in model.class_to_idx.items()}

    with torch.no_grad():
        # set to evaluation mode..
        model.eval()

        output = model.forward(image_input)
        ps = torch.exp(output)
        top_k, top_classes = ps.topk(args.top_k, dim=1)

        top_k = top_k.cpu().numpy()[0]
        top_classes = top_classes.cpu().numpy()[0]

        labels = [cat_to_name[str(index_to_class[x])] for x in top_classes]

        percentage = ['{:.5f}'.format(x) for x in top_k]

        preds = dict(zip(labels, percentage))
        return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts..')

    parser.add_argument('opts', nargs=2, type=str, help='Add the path to the image and the path to the model.')
    parser.add_argument('--gpu', dest='device', action='store_const', const='cuda',
                        default='cpu', help='Pass flag if you want to use the GPU, defaults to CPU.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='Returns top k class probabilities (default=1).')
    parser.add_argument('--category_names', default='cat_to_name.json',
                        help='Provide the label mapping JSON file (default=\'cat_to_name.json\').')

    args = parser.parse_args()

    result = predict()
    for key, val in result.items():
        print('{}\t{}'.format(key, val))

    # examples:
    # python predict.py flowers/test/19/image_06197.jpg models/2020-05-10T065735-vgg16-e10-lr0.001.pth --gpu
    # python predict.py flowers/test/19/image_06197.jpg models/2020-05-10T065735-vgg16-e10-lr0.001.pth --gpu --top_k 3
    # python predict.py flowers/test/50/image_06541.jpg models/2020-05-10T065735-vgg16-e10-lr0.001.pth --gpu --top_k 3
    # python predict.py flowers/test/102/image_08042.jpg models/2020-05-10T065735-vgg16-e10-lr0.001.pth --gpu --top_k 2
    # python predict.py flowers/test/54/image_05413.jpg models/2020-05-10T065735-vgg16-e10-lr0.001.pth --gpu --top_k 2
