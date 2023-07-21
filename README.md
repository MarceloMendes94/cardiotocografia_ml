# cardiotocografia
Dados obtidos da plataforma [UCI Machine Learning](https://archive.ics.uci.edu/)  
Link para a Base de dados [link](https://archive.ics.uci.edu/dataset/193/cardiotocography)  
Pipeline desenvolvido por [Marcelo Passamai Mendes](https://www.linkedin.com/in/marcelo-mendes/)  
## Importando a bibliotecas


```python
!pip install ipywidgets > /dev/null & echo 'Library ipywidgets Installed'
!pip install imblearn > /dev/null & echo 'Library imblearn Installed'
!pip install shap > /dev/null & echo 'Library Shap Installed'
```

    Library ipywidgets Installed
    Library imblearn Installed
    Library Shap Installed



```python
import pandas as pd
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
```

## Importando a base de dados


```python
df_description = pd.read_excel(\
                        'CTG.xls', sheet_name = 'Description',\
                         usecols = 'C:D', skiprows = 6, nrows=38,\
                         names = ['Feature','Description']      
                    )
```


```python
df = pd.read_excel(\
            'CTG.xls', sheet_name = 'Raw Data',usecols = 'D:AN')
df = df.drop(0, axis = 0)
df = df.drop([2129, 2128, 2127], axis = 0)
```

## FAST EDA
**Exploratory data analysis**  
Biblioteca usada [ydata-profiling](https://github.com/ydataai/ydata-profiling) 


```python
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("report.html")
```

Foi realizado uma análise incial usando a biblioteca ydata_profiling, no link a seguir:  
[ydata_report](report.html)  

### considerações 
1. Baixa variabilidade  
As colunas DS,DP,DR tem variabilidade menor que 1% sendo muito baixo podendo ser considerado constante, serão removidas por isso.  

2. Alta correlação  
Coluna CLASS tem uma correlação muito alta com as colunas A,B,C,D,E,AD,DE,FS,SUSP e NSP, por esse motivo será removida. Além disso, há uma correlação forte entre as colunas: LB e LBE, representando o baseline do sistema (sisporto) e (expecialista), porém não serão removidas.  

3. Balanceamento de carga  
Existe desbalanceamento de carga muito alto nacoluna alvo NSP por esse motivo recomento uma estratégia de balanceamento de carga antes da construção do modelo.



```python
df = df.drop(['CLASS','DR','DS','DP'], axis = 1)
```

## Pipeline

https://lerekoqholosha9.medium.com/random-oversampling-and-undersampling-for-imbalanced-classification-a4aad406fd72


```python
X = df.drop('NSP', axis = 1)
y = df['NSP']
```


```python
ros = RandomOverSampler(sampling_strategy = {2.0:500,3.0:500}, random_state = 42)
```


```python
rus = RandomUnderSampler(sampling_strategy = 'majority', random_state = 42)
```


```python
# Decision tree
param_dt = {'criterion':['gini', 'entropy', 'log_loss'],\
           'max_depth':[2,3,4,5,7],}

# Random Forest
params_rf = {'n_estimators':[100,150,200,250,300,700],\
             'max_depth':[2,3,4,5,6,7,],\
             'criterion':['gini', 'entropy', 'log_loss'],}


models = [ (DecisionTreeClassifier(), param_dt, 'Decision Tree'),\
           (RandomForestClassifier(), params_rf, 'Random Forest')]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
cv = 5
df_result = pd.DataFrame()
classifier_list = []

for model_ in models:
    grid = GridSearchCV(model_[0], param_grid = model_[1], cv = cv,\
                        n_jobs = -1, scoring = 'accuracy')
    pipe = Pipeline(
        [('Rand_Over_Sample',ros),\
         ('Rand_Under_Sample',rus),\
         ('scaler', StandardScaler()),\
         (model_[2], grid)])

    pipe.fit(X_train, y_train)
    #pipe.score(X_test, y_test)
    index = pipe[model_[2]].best_index_
    cv_results = pipe[model_[2]].cv_results_
    df_result[model_[2]] = [cv_results[f"split{i}_test_score"][index] for i in range(cv)]
    
    # Salvo para acessar o modelo desejado
    # criado pelo pipeline em caso de uso
    # em produção ou para uso de explicabilidade como o shap
    #classifier_list['Decision Tree'].best_estimator_
    #classifier_list.predict(X_test)
    classifier_list.append(pipe)
```


```python
def boxplot_sorted(df, score, title, rot=90, figsize=(10,6), fontsize=12):
    df2 = df
    meds = df2.median().sort_values(ascending=False)
    axes = df2[meds.index].boxplot(figsize=figsize, rot=rot, fontsize=fontsize,
                                   boxprops=dict(linewidth=4, color='cornflowerblue'),
                                   whiskerprops=dict(linewidth=4, color='cornflowerblue'),
                                   medianprops=dict(linewidth=4, color='firebrick'),
                                   capprops=dict(linewidth=4, color='cornflowerblue'),
                                   flierprops=dict(marker='o', markerfacecolor='dimgray',
                                        markersize=12, markeredgecolor='black'),
                                   return_type="axes")
    axes.set_title(title, fontsize=fontsize)
    plt.savefig(title + '.pdf')
    plt.show()
```


```python
boxplot_sorted(df_result,'Accuracy','Boxplot of models RF and DT in Accuracy')
```


    
![png](output_19_0.png)
    



```python
classifier_list[1]['Random Forest'].best_estimator_
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestClassifier(criterion=&#x27;log_loss&#x27;, max_depth=7, n_estimators=200)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier(criterion=&#x27;log_loss&#x27;, max_depth=7, n_estimators=200)</pre></div></div></div></div></div>






