"""
README for B.A.B.E.L Backend
"""

# B.A.B.E.L Backend API

API para execução do modelo GPT e comunicação com hardware (óculos AR) via Tailscale.

## Setup

### 1. Instale dependências

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure variáveis de ambiente

```bash
cp .env.example .env
# Edite .env conforme necessário
```

### 3. Inicie o servidor

```bash
# Opção 1: Script bash
bash run.sh

# Opção 2: Direto
python main.py

# Opção 3: Com uvicorn
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

O servidor estará disponível em: `http://localhost:8000`

## 📖 Documentação da API

A documentação interativa está disponível em:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Endpoints da API

### Health Check
```
GET /health
```
Verifica se o backend está funcionando.

### Model Info
```
GET /info
```
Retorna informações do modelo carregado.

### Gerar Texto
```
POST /generate
Content-Type: application/json

{
  "prompt": "Hello world",
  "max_tokens": 256,
  "temperature": 0.7,
  "top_k": 50
}
```

### Gerar Lote
```
POST /generate-batch
Content-Type: application/json

[
  {"prompt": "texto 1"},
  {"prompt": "texto 2"}
]
```

### Codificar Texto
```
POST /encode
Content-Type: application/json

{
  "text": "Hello world"
}
```

### Decodificar Tokens
```
POST /decode
Content-Type: application/json

{
  "tokens": [1, 2, 3, 4, 5]
}
```

### Estatísticas
```
POST /stats
```
Retorna métricas de desempenho e uso.

## Conexão via Tailscale

Para conectar o hardware (óculos AR) ao backend via Tailscale:

1. **Instale Tailscale** em ambas as máquinas
2. **Autentique** cada máquina na sua rede Tailscale
3. **Use o IP da máquina backend** na configuração do hardware

Exemplo com IP Tailscale `100.x.x.x`:
```bash
curl http://100.x.x.x:8000/health
```

## Cliente Python

Use o cliente fornecido para comunicação fácil:

```python
from backend.client import BackendClientSync

# Conectar ao backend (pode ser IP Tailscale)
client = BackendClientSync(base_url="http://100.x.x.x:8000")

# Gerar texto
result = client.generate("Olá, qual é o seu nome?")
print(result['generated_text'])
```

## Desenvolvimento

Para testes locais sem validação:

```bash
# O servidor aceita qualquer request sem validação rigorosa
# Útil para testar com hardware diferente
```

## Docker

```bash
docker build -t babel-backend .
docker run -p 8000:8000 babel-backend
```

## Performance

- **Device**: Detecta automaticamente GPU (CUDA) ou CPU
- **Max Tokens**: Configurável por request
- **Batch**: Suporte para múltiplos requests simultâneos

## TODOs

- [ ] Implementar tokenizer real (BPE)
- [ ] Adicionar autenticação e validação
- [ ] Melhorar tratamento de erros
- [ ] Adicionar logging estruturado
- [ ] WebSocket para streaming de resposta
- [ ] Rate limiting
