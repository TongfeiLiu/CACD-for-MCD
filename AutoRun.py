import os

# for ps in range(3, 23, 2):
#     path = './vision/ps_{}/'.format(ps)
#     if not os.path.exists(path):
#         os.makedirs(path)
#     os.system('python train.py --patch_size={} --vision_path={}'.format(ps, path))
for tps in range(7, 15, 4):
    pth = './vision/patch_size{}/'.format(tps)
    if not os.path.exists(pth):
        os.makedirs(pth)
    for ps in range(1, 2, 1):
        print('-------- patch_size-{}: iter-{} ---------\n'.format(tps, ps))
        path = './vision/patch_size{}/iter_{}/'.format(tps, ps)
        if not os.path.exists(path):
            os.makedirs(path)
        os.system('python train.py --patch_size={} --vision_path={}'.format(tps, path))
