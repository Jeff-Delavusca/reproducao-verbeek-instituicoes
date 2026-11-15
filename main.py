import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# ==============================================================================
# CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================
# Substitua 'dados_verbeek.dta' pelo nome/caminho do seu arquivo de dados (.dta ou .csv)
# df = pd.read_stata("dados_verbeek.dta")
# df = pd.read_csv("dados.csv")

# Garantir que os nomes das colunas fiquem em minúsculas
# df.columns = df.columns.str.lower()


# ==============================================================================
# 1. TABELA 5.4: OLS Results Explaining GDP per capita
# ==============================================================================
def rodar_tabela_54(df):
    print("\n" + "=" * 75)
    print("Table 5.4 OLS results explaining GDP per capita")
    print("Dependent variable: log(GDP)")
    print("=" * 75)
    
    d1 = df.dropna(subset=['loggdp', 'qi', 'latitude'])
    m1 = smf.ols('loggdp ~ qi + latitude', data=d1).fit()
    
    d2 = df.dropna(subset=['loggdp', 'qi', 'latitude', 'africa', 'asia', 'malfal94'])
    m2 = smf.ols('loggdp ~ qi + latitude + africa + asia + malfal94', data=d2).fit()
    
    vars_order = ['Intercept', 'qi', 'latitude', 'africa', 'asia', 'malfal94']
    models = {'(1)': m1, '(2)': m2}
    
    out = []
    for var in vars_order:
        var_name = 'const' if var == 'Intercept' else var
        row = {'Variável': var_name}
        for name in ['(1)', '(2)']:
            m = models[name]
            if var in m.params:
                row[f"{name} Coef"] = f"{m.params[var]:.3f}"
                row[f"{name} DP"] = f"({m.bse[var]:.3f})"
            else:
                row[f"{name} Coef"] = "–"
                row[f"{name} DP"] = ""
        out.append(row)
        
    df_res = pd.DataFrame(out)
    print(df_res.to_string(index=False))
    print("-" * 75)
    print(f"R²                            | {m1.rsquared:.3f}         | {m2.rsquared:.3f}")
    print(f"Número de observações          | {int(m1.nobs)}            | {int(m2.nobs)}")
    print("=" * 75)


# ==============================================================================
# 2. TABELA 5.5: OLS Results Reduced Form (QI Explained from Exogenous Vars)
# ==============================================================================
def rodar_tabela_55(df):
    print("\n" + "=" * 75)
    print("Table 5.5 OLS results reduced form (QI explained from exogenous variables)")
    print("Dependent variable: QI")
    print("=" * 75)
    
    specs = [
        ('(1a)', ['logem4', 'latitude'], ['logem4 = 0']),
        ('(1b)', ['logem4', 'euro1900', 'latitude'], ['logem4 = 0', 'euro1900 = 0']),
        ('(2a)', ['logem4', 'latitude', 'africa', 'asia', 'malfal94'], ['logem4 = 0']),
        ('(2b)', ['logem4', 'euro1900', 'latitude', 'africa', 'asia', 'malfal94'], ['logem4 = 0', 'euro1900 = 0'])
    ]
    
    models = {}
    f_tests = {}
    
    for name, rhs, f_hyp in specs:
        d = df.dropna(subset=['qi'] + rhs)
        formula = f"qi ~ {' + '.join(rhs)}"
        m = smf.ols(formula, data=d).fit()
        models[name] = m
        # Teste F exclusivo sobre a relevância dos instrumentos no 1º estágio
        f_val = float(m.f_test(f_hyp).statistic)
        f_tests[name] = f_val

    all_vars = ['Intercept', 'logem4', 'euro1900', 'latitude', 'africa', 'asia', 'malfal94']
    out = []
    for var in all_vars:
        var_name = 'const' if var == 'Intercept' else var
        row = {'Variável': var_name}
        for name in ['(1a)', '(1b)', '(2a)', '(2b)']:
            m = models[name]
            if var in m.params:
                row[f"{name} Coef"] = f"{m.params[var]:.3f}"
                row[f"{name} DP"] = f"({m.bse[var]:.3f})"
            else:
                row[f"{name} Coef"] = "–"
                row[f"{name} DP"] = ""
        out.append(row)
        
    df_res = pd.DataFrame(out)
    print(df_res.to_string(index=False))
    print("-" * 75)
    print("Testes e Estatísticas:")
    f_row = "F-test dos instrumentos       | " + " | ".join([f"{f_tests[n]:.2f}" for n in ['(1a)', '(1b)', '(2a)', '(2b)']])
    r2_row = "R²                            | " + " | ".join([f"{models[n].rsquared:.3f}" for n in ['(1a)', '(1b)', '(2a)', '(2b)']])
    n_row = "Número de observações          | " + " | ".join([f"{int(models[n].nobs)}" for n in ['(1a)', '(1b)', '(2a)', '(2b)']])
    print(f_row)
    print(r2_row)
    print(n_row)
    print("=" * 75)


# ==============================================================================
# 3. TABELA 5.6: IV Results Explaining GDP per capita
# ==============================================================================
def estimar_iv_2sls(df, dep_var, exog_vars, endog_var, inst_vars):
    todas_vars = list(dict.fromkeys([dep_var, endog_var] + exog_vars + inst_vars))
    dados = df.dropna(subset=todas_vars).copy()
    N = len(dados)
    
    # 1º Estágio
    X_1st = sm.add_constant(dados[exog_vars + inst_vars])
    mod_1st = sm.OLS(dados[endog_var], X_1st).fit()
    dados['endog_hat'] = mod_1st.fittedvalues
    dados['v_hat'] = mod_1st.resid  # Resíduo da forma reduzida
    
    # 2º Estágio (Matrizes usando .values)
    X_2nd_fit = sm.add_constant(dados[exog_vars + ['endog_hat']])
    X_2nd_act = sm.add_constant(dados[exog_vars + [endog_var]])
    y = dados[dep_var].values
    
    X_fit_mat = X_2nd_fit.values
    X_act_mat = X_2nd_act.values
    
    # Coeficientes do MQO2E: beta = (X_hat' X_hat)^(-1) X_hat' y
    beta = np.linalg.inv(X_fit_mat.T @ X_fit_mat) @ X_fit_mat.T @ y
    
    # Resíduos estruturais reais e erros-padrão homocedásticos com ajuste N - K
    e_struct = y - X_act_mat @ beta
    K = X_act_mat.shape[1]
    s2 = np.sum(e_struct**2) / (N - K)
    cov_beta = s2 * np.linalg.inv(X_fit_mat.T @ X_fit_mat)
    se_beta = np.sqrt(np.diag(cov_beta))
    
    varnames = [c if c != 'endog_hat' else endog_var for c in X_2nd_fit.columns]
    params = pd.Series(beta, index=varnames)
    bse = pd.Series(se_beta, index=varnames)
    
    # Teste de Durbin–Wu–Hausman (t-stat do v_hat no OLS ampliado)
    exog_formula = (' + ' + ' + '.join(exog_vars)) if len(exog_vars) > 0 else ''
    formula_dwh = f'{dep_var} ~ {endog_var}{exog_formula} + v_hat'
    mod_dwh = smf.ols(formula_dwh, data=dados).fit()
    dwh_t = mod_dwh.tvalues['v_hat']
    dwh_p = mod_dwh.pvalues['v_hat']
    
    # Teste de Sargan (Apenas em modelos sobreidentificados)
    num_sobreid = len(inst_vars) - 1
    if num_sobreid > 0:
        mod_sargan = sm.OLS(e_struct, X_1st).fit()
        sargan_stat = N * mod_sargan.rsquared
        sargan_p = 1 - chi2.cdf(sargan_stat, num_sobreid)
    else:
        sargan_stat, sargan_p = None, None
        
    return {
        'params': params,
        'bse': bse,
        'N': N,
        'inst_names': ", ".join(inst_vars),
        'dwh_t': dwh_t,
        'dwh_p': dwh_p,
        'sargan_stat': sargan_stat,
        'sargan_p': sargan_p
    }

def rodar_tabela_56(df):
    print("\n" + "=" * 75)
    print("Table 5.6 IV results explaining GDP per capita")
    print("Dependent variable: log(GDP)")
    print("=" * 75)
    
    m1a = estimar_iv_2sls(df, 'loggdp', ['latitude'], 'qi', ['logem4'])
    m1b = estimar_iv_2sls(df, 'loggdp', ['latitude'], 'qi', ['logem4', 'euro1900'])
    m2a = estimar_iv_2sls(df, 'loggdp', ['latitude', 'africa', 'asia', 'malfal94'], 'qi', ['logem4'])
    m2b = estimar_iv_2sls(df, 'loggdp', ['latitude', 'africa', 'asia', 'malfal94'], 'qi', ['logem4', 'euro1900'])
    
    models = {'(1a)': m1a, '(1b)': m1b, '(2a)': m2a, '(2b)': m2b}
    vars_order = ['const', 'qi', 'latitude', 'africa', 'asia', 'malfal94']
    
    out = []
    for var in vars_order:
        row_name = 'QI (instrumented)' if var == 'qi' else var
        row = {'Variável': row_name}
        for name in ['(1a)', '(1b)', '(2a)', '(2b)']:
            m = models[name]
            if var in m['params']:
                row[f"{name} Coef"] = f"{m['params'][var]:.3f}"
                row[f"{name} DP"] = f"({m['bse'][var]:.3f})"
            else:
                row[f"{name} Coef"] = "–"
                row[f"{name} DP"] = ""
        out.append(row)
        
    df_res = pd.DataFrame(out)
    print(df_res.to_string(index=False))
    print("-" * 75)
    print("Instrumentos e Testes de Especificação:")
    print("Instruments                   | " + " | ".join([models[n]['inst_names'] for n in ['(1a)', '(1b)', '(2a)', '(2b)']]))
    
    sargan_str = []
    for n in ['(1a)', '(1b)', '(2a)', '(2b)']:
        st = models[n]['sargan_stat']
        pv = models[n]['sargan_p']
        if st is not None:
            sargan_str.append(f"{st:.3f} (p={pv:.3f})")
        else:
            sargan_str.append("–")
            
    dwh_str = [f"{models[n]['dwh_t']:.2f} (p={models[n]['dwh_p']:.3f})" for n in ['(1a)', '(1b)', '(2a)', '(2b)']]
    n_str = [str(models[n]['N']) for n in ['(1a)', '(1b)', '(2a)', '(2b)']]
    
    print("Overidentifying restrictions   | " + " | ".join(sargan_str))
    print("Durbin–Wu–Hausman test (t)    | " + " | ".join(dwh_str))
    print("Número de observações          | " + " | ".join(n_str))
    print("=" * 75)


if __name__ == "__main__":
    # Exemplo de uso após carregar df:
    # df = pd.read_stata("dados_verbeek.dta")
    # rodar_tabela_54(df)
    # rodar_tabela_55(df)
    # rodar_tabela_56(df)
    pass
