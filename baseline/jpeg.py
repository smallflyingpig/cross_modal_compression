"""
read images and save the with different QF
"""
import imageio, argparse, os, glob, tqdm


def get_parser():
    parser = argparse.ArgumentParser("compress images with JPEG")
    parser.add_argument("--input_folder", type=str, default="", help="")
    parser.add_argument("--output_folder", type=str, default="", help="")
    parser.add_argument("--QF", type=int, default=30, help="")
    parser.add_argument("--ext_list", type=str, default="jpg,jpeg,png", help="")
    parser.add_argument("--save_format", type=str, choices=['JPEG-FI'], default='JPEG-FI', help="see https://imageio.readthedocs.io/en/stable/formats.html for details")
    args = parser.parse_args()
    return args

def main(args):
    input_folder, output_folder = args.input_folder, args.output_folder
    if input_folder == "":
        print("please provide the input folder")
        return
    if output_folder == "":
        print("please provide the output folder")
        return
    if not os.path.exists(args.output_folder):
        print("the output folder does not exist, make it")
        os.makedirs(args.output_folder)

    file_all = []
    ext_list = args.ext_list.split(',')
    for folder, path, files in os.walk(input_folder):
        for _f in files:
            if os.path.splitext(_f)[-1][1:] in ext_list:
                file_all.append(os.path.join(folder, _f))

    # for ext in args.ext_list.split(','):
    #     file_this = glob.glob(os.path.join(input_folder,'*.'+ext))
    #     file_all += file_this
    #     print("file number with ext {}:{}".format(ext, len(file_this)))
    print("file total: ", len(file_all))
    for _f in tqdm.tqdm(file_all):
        # print(_f)
        img = imageio.imread(_f)
        filename = os.path.split(os.path.splitext(_f)[0])[-1]
        output_filename = os.path.join(output_folder, filename+'.jpeg')
        # print(output_filename)
        imageio.imwrite(output_filename, img, args.save_format, quality=args.QF)
    print("compress images end. save them to {}, file total {}".format(output_folder, len(file_all)))

    

if __name__=="__main__":
    imageio.plugins.freeimage.download()
    args = get_parser()
    main(args)
