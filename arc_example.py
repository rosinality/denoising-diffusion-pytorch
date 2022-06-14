
import cv2
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from recog_backbones import get_recog


# Arc 입력 Scaling
transformer_Arcface = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])

# ArcFace 모델로드
Arc_path       = 'embedder_model/partial_fc_glint360k_r100.pth'
resnet_name    = Arc_path.split('_')[-1].split('.')[0]
embedder       = get_recog(resnet_name, fp16=False) # get_recog 함수에서 r100, r50 등의 입력으로 모델 구조를 결정
embedder.load_state_dict(torch.load(Arc_path))

# GPU 설정
device         = 'cuda'
embedder       = embedder.to(device)
embedder.eval()


# 이미지 로드 및 처리 후 Arc Embedder 출력 획득
img_id         = cv2.imread('/data/image/images256x256/00030907.jpg')
img_id         = Image.fromarray(img_id)
img_id         = transformer_Arcface(img_id)
img_id         = img_id.view(-1, img_id.shape[0], img_id.shape[1], img_id.shape[2]).cuda()
img_id         = img_id.cuda()
img_id_01      = (img_id + 1) / 2
img_id_01_down = F.interpolate(img_id_01, scale_factor=112/img_id_01.shape[-1])
latent_id      = embedder(img_id_01_down)
latent_id      = F.normalize(latent_id, p=2, dim=1)
