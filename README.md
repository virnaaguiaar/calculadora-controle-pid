# Calculadora de Controle (Streamlit)

Projeto de PD, PI, PID (zeros reais e iguais) e a/(s+b) pelo lugar das raízes,
com verificação em malha fechada e discretização (Tustin, Euler, ZOH).

## Arquivos

- `app.py`: a interface (o que aparece no navegador)
- `controle.py`: os cálculos (critérios de ângulo e módulo, discretização)
- `requirements.txt`: as bibliotecas que o site precisa instalar

## Rodar no computador (Ubuntu)

```bash
cd ~/projeto_controle
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Abre em http://localhost:8501. Para parar: Ctrl+C no terminal.

## Publicar na internet (Streamlit Community Cloud, grátis)

1. Crie uma conta em https://github.com e um repositório público, por exemplo `calculadora-controle`.
2. No terminal, dentro da pasta do projeto:

```bash
git init
git add app.py controle.py requirements.txt README.md .gitignore
git commit -m "Calculadora de controle"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/calculadora-controle.git
git push -u origin main
```

3. Entre em https://share.streamlit.io com a conta do GitHub.
4. Clique em **Create app**, escolha o repositório, a branch `main` e o arquivo `app.py`.
5. Clique em **Deploy**. Em alguns minutos o site fica no ar com um link `.streamlit.app`.

Para atualizar o site depois: edite, rode `git add .`, `git commit -m "..."` e `git push`.
O Streamlit Cloud reimplanta sozinho.
