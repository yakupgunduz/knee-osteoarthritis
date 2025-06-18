import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Önbellek dizinini değiştir
import os
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

# Modelin cihaz ayarını yap
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Modeli oluştur ve ağırlıkları yükle
model_path = model_path  # Model dosya yolu
model = deeplabv3_mobilenet_v3_large(pretrained=False, aux_loss=True)
model.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)

# Modeli değerlendirme moduna al
model.eval()

# Görüntü nesnesini işleyerek tahmin yap
image = image.convert("RGB")

input_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

# Model tahmini
with torch.no_grad():
    output = model(input_tensor)['out']
    output = torch.sigmoid(output).squeeze().cpu().numpy()

# Maskeyi oluştur
output_mask = (output > 0.5).astype('uint8') * 255

# Maskeyi PIL.Image nesnesine dönüştür
output_image = Image.fromarray(output_mask)

# Maskeyi görselleştirme (isteğe bağlı)
plt.imshow(output_image, cmap="gray")
plt.show()

# Çıktılar
output = output_image  # PIL.Image nesnesi olarak tanımlandı
points_list = [[40, 50], [50, 60], [60, 70], [70, 80]]