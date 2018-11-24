import os

path = '/home/ubuntu/PycharmProjects/ALISURE/opticalflow/flownet2-pytorch/work2/run'
png_path = '/home/ubuntu/PycharmProjects/ALISURE/opticalflow/flownet2-pytorch/work2/run-png'
flo_files = os.listdir(path)

if not os.path.exists(png_path):
    os.makedirs(png_path)
    pass

for flo_file_index, flo_file in enumerate(flo_files):

    ml = './C/color_flow ' + os.path.join(path, flo_file) + " " + os.path.join(
        png_path, "{}.png".format(os.path.splitext(flo_file)[0]))
    print(ml)
    os.system(ml)
    pass
