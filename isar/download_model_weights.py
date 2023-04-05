import gdown

gdown.download('https://drive.google.com/uc?id=1qKhp_6-PQOk7ONhOScObDG1VCHCpSpVl&export=download',
    './modelzoo/u2net.pth',
    quiet=False)

gdown.download('https://drive.google.com/uc?id=1HS_rLIN3JBJZaN2CWPFM43xpnDwqSC_U&export=download',
    './modelzoo/dino_resnet50_pretrain.pth',
    quiet=False)

gdown.download('https://drive.google.com/uc?id=1ePMF73_mJpkTjspaGeB6DoAvLTVzEGR4&export=download',
    './exps/checkpoint0159.pth',
    quiet=False)

gdown.download('https://drive.google.com/uc?id=1wGnKVU7QebJnHzzI7SBKFpQmhxnPqMt4&export=download',
    './exps/checkpoint0104.pth',
    quiet=False)

gdown.download('https://drive.google.com/uc?id=12tygFzGxYiuGxWX4HpKhY2lT2HaMvk3S&export=download',
    './exps/checkpoint0099.pth',
    quiet=False)