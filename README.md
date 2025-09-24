# GAN — Gerador de Dígitos (MNIST)

Um projeto de IA Generativa que treina e disponibiliza um modelo GAN (Generative Adversarial Network) para criar imagens realistas de dígitos manuscritos (0–9) a partir do dataset MNIST. O repositório inclui um app Streamlit com design premium para exploração do modelo, visualização da evolução do treinamento e geração interativa de amostras.

• Autor/Mantenedor: [@sidnei-almeida](https://github.com/sidnei-almeida)

## Visão Geral

Este projeto foi desenvolvido para:
- Treinar um modelo GAN para MNIST (28×28 px, escala de cinza)
- Salvar checkpoints do treinamento para auditoria e reprodutibilidade
- Registrar e exibir imagens de evolução do gerador ao longo das épocas
- Disponibilizar um app Streamlit com interface dark e paleta fria premium, com:
  - Geração interativa (grade 4×4/5×5/6×6, seed e latent dimension)
  - Métricas visuais do lote gerado (diversidade, esparsidade, assimetria)
  - Travessia do espaço latente (latent traversal)
  - Galeria das imagens por checkpoints (ex.: a cada 50 épocas)
  - Leitura e visualização dos arquivos de `training_checkpoints/`

## Estrutura do Repositório

```
.
├── app.py                         # App Streamlit (GAN)
├── requirements.txt               # Dependências do app GAN
├── run_app_gan.sh                 # Script para executar o app
├── imagens/                       # Imagens geradas por checkpoints (evolução visual)
│   └── image_for_ckpt_*.png
├── training_checkpoints/          # Checkpoints do treinamento (TensorFlow)
│   ├── checkpoint
│   └── ckpt-<epoca>-<etapa>.*
├── modelos/
│   └── generator_model.keras      # Modelo gerador salvo (Keras)
├── notebooks/                     # Notebooks com EDA e construção do modelo
│   ├── 1_Exploratory_Data_Analysis.ipynb
│   └── 2_Model_Contruction.ipynb
└── keras_imdb_sentiment_lstm/     # Projeto paralelo (app IMDB LSTM) — referência de design
```

## Pré-requisitos

- Python 3.10+ (recomendado 3.10–3.11)
- Pip e venv (ou conda)
- Ambiente Linux/macOS/WSL (Windows nativo também funciona, mas TensorFlow GPU exige setup adicional)

## Instalação e Execução do App

1) Clone o repositório
```bash
git clone https://github.com/sidnei-almeida/gan_gerador_digitos_mnist.git
cd gan_gerador_digitos_mnist
```

2) Crie e ative um ambiente virtual (opcional, recomendado)
```bash
python3 -m venv venv
source venv/bin/activate
```

3) Instale as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4) Execute o app Streamlit
```bash
chmod +x run_app_gan.sh
./run_app_gan.sh
```

O app iniciará em: http://localhost:8501

## Uso do App

- Início: visão geral, status do sistema e evolução visual do treinamento
- Modelo: resumo do `generator_model.keras` (shape de saída e nº de parâmetros)
- Checkpoints: lista e gráfico de progresso com base em `training_checkpoints/`
- Imagens: galeria filtrável por intervalo de épocas (ex.: 50 em 50)
- Geração: criação de grades com parâmetros (latent dim, seed), métricas do lote e travessia de latente

Para a página de Geração funcionar, certifique-se de ter o arquivo do gerador salvo em pelo menos um dos caminhos:
- `modelos/generator_model.keras`
- `notebooks/generator_model.keras` (fallback)

## Dados de Evolução e Checkpoints

- A pasta `imagens/` contém os snapshots do gerador em diferentes checkpoints (ex.: `image_for_ckpt_0.png`, `image_for_ckpt_50.png`, ...). O app usa esses arquivos para mostrar a evolução visual.
- A pasta `training_checkpoints/` contém os checkpoints do TensorFlow (`ckpt-*.index`, `ckpt-*.data-00000-of-00001`) e o arquivo `checkpoint` com metadados. O app lê esses nomes/arquivos para montar um gráfico simplificado de progresso.

## Notebooks

Os notebooks em `notebooks/` incluem:
- `1_Exploratory_Data_Analysis.ipynb`: análise exploratória do dataset MNIST
- `2_Model_Contruction.ipynb`: construção/treino do modelo GAN (arquitetura, hiperparâmetros e salvamento de checkpoints)

## Dicas e Solução de Problemas

- Caso apareça erro ao carregar o modelo, verifique o caminho do arquivo `generator_model.keras`.
- Se as imagens não forem listadas, revise o padrão dos nomes: `imagens/image_for_ckpt_<numero>.png`.
- Em ambientes sem GPU, o app funciona normalmente, apenas a etapa de treino (se executada) será mais lenta.
- Para TensorFlow com GPU (opcional), siga a documentação oficial da sua versão de CUDA/cuDNN.

## Licença

Este projeto é disponibilizado para fins educacionais e de demonstração. Adapte conforme necessário para uso comercial. Verifique licenças de dependências de terceiros.

## Autor

Criado e mantido por [Sidnei Almeida](https://github.com/sidnei-almeida).

Contato: <sidnei.almeida1806@gmail.com>
