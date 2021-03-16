from train_u2 import *


def infer_and_save_single_tif(img_path, model_path, output_img_path):
    '''
    Makes a prediction using the current model on a single input image
    :param inL: int, UNETs input image size [px]
    :param outL: int, UNETs output size [px]
    :param mask: binary 2D np.array, mask of the OD image with zeros on the target area
    :param model: model, the trained UNET model
    :param img_path: str, path for the input image
    :return:
    X: np.array, the masked input image
    Y_prime: np.array, the predicted image
    Y: np.array, the target image (or the unmasked input image)
    '''

    parser = get_parser()
    args = parser.parse_args()

    outL = 2 * args.maskR  # Output size
    mask = generate_mask(args.inL, args.maskR)
    model = u2net_2d((args.inL,args.inL,1),1,[64, 128, 256, 512])

    model.load_weights(model_path)
    _, y_prime, _ = infer_single_img(args.inL, outL, mask, model, img_path)
    save_png(output_img_path, y_prime)
    #save_tif(output_img_path, y_prime)

    return

if __name__ == '__main__':

    model_number = '0740'
    modelFile = './models/u2net_epoch_{}.h5'.format(model_number)
    img_path = './data/small_train_cropped256px/000002.png'
    output_img_path = './result/L2loss_epoch{0}_2.png'.format(model_number)

    infer_and_save_single_tif(img_path, modelFile, output_img_path)



