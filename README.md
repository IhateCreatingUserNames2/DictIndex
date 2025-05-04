# Índice de Ditador
DEMO: https://dictindex.onrender.com/
####
Um site que monitora e analisa notícias relacionadas a Donald Trump para avaliar possíveis tendências autoritárias e gerar um índice de autoritarismo com base nessas análises.

## Descrição

O "Índice de Ditador" é uma ferramenta que:

1. Busca notícias recentes sobre Donald Trump usando a API OpenAI com o modelo gpt-4o-search-preview
2. Analisa cada notícia para identificar possíveis tendências autoritárias usando critérios específicos
3. Atribui uma pontuação de 0 a 10 para cada notícia, onde 10 representa tendências altamente autoritárias
4. Calcula um índice geral com base na média das pontuações de todas as notícias analisadas
5. Exibe as notícias analisadas com suas respectivas pontuações e justificativas

## Pré-requisitos

- Python 3.8 ou superior
- Conta OpenAI com acesso à API
- Chave de API da OpenAI

## Instalação

1. Clone este repositório:
   ```
   git clone https://github.com/IhateCreatingUserNames2/DictIndex/
   cd indice-ditador
   ```

2. Instale as dependências:
   ```
   pip install fastapi uvicorn httpx
   ```

3. Configure sua chave de API:
   - Opção 1: Configure uma variável de ambiente `OPENAI_API_KEY`
   - Opção 2: Adicione sua chave na interface web após iniciar o aplicativo

## Uso

1. Inicie o servidor:
   ```
   uvicorn app:app --reload
   ```

2. Acesse a interface web em `http://localhost:8000`

3. Na interface web:
   - Configure sua chave de API OpenAI (se necessário)
   - Clique em "Atualizar Notícias" para buscar e analisar notícias recentes
   - Visualize o "Índice de Ditador" atual e os detalhes das notícias analisadas

## Estrutura do Projeto

- `app.py` - Backend FastAPI que gerencia a lógica do servidor, a busca de notícias e a análise de tendências autoritárias
- `static/index.html` - Interface do usuário para visualizar o índice e as notícias
- `codebase.txt` - Arquivo de banco de dados que armazena notícias analisadas e suas pontuações

## Metodologia

O índice é calculado com base nos seguintes critérios:

1. Retórica que ataca instituições democráticas
2. Ameaças à liberdade de imprensa ou oponentes políticos
3. Abuso de poder executivo
4. Tentativas de minar a independência judicial
5. Desrespeito às normas e processos democráticos

## Implantação

Este aplicativo pode ser implantado facilmente no Render:

1. Crie uma nova conta ou faça login em https://render.com
2. Selecione "New Web Service"
3. Conecte o repositório do GitHub
4. Configure o serviço:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Adicione a variável de ambiente `OPENAI_API_KEY` nas configurações (opcional)

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorar o projeto.

## Licença

Este projeto está licenciado sob a licença MIT. V
