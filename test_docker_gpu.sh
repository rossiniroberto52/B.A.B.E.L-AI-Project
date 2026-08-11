#!/bin/bash
IMAGEM="${1:-jrcichra/rocm-pytorch-gfx803}"
echo "Testando imagem: $IMAGEM"

docker run -it --rm -v $(pwd):/projects --privileged --name pytorch_test \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  "$IMAGEM" \
  bash -c "
    # Instala PyTorch se não estiver presente
    python3 -c 'import torch' 2>/dev/null || {
      echo 'PyTorch não encontrado, instalando...'
      pip install torch --index-url https://download.pytorch.org/whl/rocm6.1.2
    }

    cd /tmp && python3 -c \"
import torch
print('PyTorch version:', torch.__version__)
print('CUDA disponível:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
a = torch.randn(100, device='cuda')
b = torch.randn(100, device='cuda')
c = a + b
print('Soma simples OK:', c.sum().item())
\"
"
