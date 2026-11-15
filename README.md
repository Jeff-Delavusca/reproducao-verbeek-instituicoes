# Reprodução Econométrica: Instituições e Desenvolvimento Econômico

Este repositório contém os códigos em Python, relatórios, planilhas e arquivos de log do Stata referentes à reprodução empírica da **Seção 5.7** do livro *A Guide to Modern Econometrics* (Marno Verbeek, 5ª edição, 2017).

A análise reproduz a estratégia de identificação econométrica do artigo clássico de **Acemoglu, Johnson e Robinson (2001)** (*"The Colonial Origins of Comparative Development: An Empirical Investigation"*), investigando a relação causal entre a qualidade das instituições e o desenvolvimento econômico de longo prazo (PIB *per capita*) através do método de **Variáveis Instrumentais (MQO2E / 2SLS)**.

---

## 📂 Estrutura do Repositório

```text
reproducao-verbeek-instituicoes/
├── README.md                        # Documentação do repositório
├── requirements.txt                 # Dependências do Python
├── main.py                          # Script Python com a estimação de todos os modelos
├── data/
│   └── dados_verbeek.xlsx            # Conjunto de dados (Acemoglu, Johnson & Robinson, 2001)
└── outputs/
    ├── reproducao_verbeek_5_7.xlsx  # Planilha formatada com as Tabelas 5.4, 5.5 e 5.6
    └── Relatorio_Reproducao_Verbeek_5_7.docx # Relatório acadêmico completo em Word
