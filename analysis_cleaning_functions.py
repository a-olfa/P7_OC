#analysis_cleaning_function.py >
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import scipy.stats as stats
import inspect

import gc
def group_by_agregate(df,  column_new, column_origin, column_grouped, agragation) :
    if column_origin in get_numerical_categorical(df)[0]:
        if agragation =='sum':
            df[column_new] = df[column_origin]
            grp = df.groupby(by = column_grouped)[column_origin].sum().reset_index().rename(index = str, columns = {column_new :column_origin})
            df = df.merge(grp, on = column_grouped, how = 'left')
            del grp
            gc.collect()
        elif agragation =='mean':
            df[column_new] = df[column_origin]
            grp = df.groupby(by = column_grouped)[column_origin].mean().reset_index().rename(index = str, columns = {column_new : column_origin})
            df = df.merge(grp, on = column_grouped, how = 'left')
            del grp
            gc.collect()
        elif agragation =='max':
            df[column_new] = df[column_origin]
            grp = df.groupby(by = column_grouped)[column_origin].max().reset_index().rename(index = str, columns = {column_new : column_origin})
            df = df.merge(grp, on = column_grouped, how = 'left')
            del grp
            gc.collect()
        elif agragation =='min':
            df[column_new] = df[column_origin]
            grp = df.groupby(by = column_grouped)[column_origin].min().reset_index().rename(index = str, columns = {column_new : column_origin})
            df = df.merge(grp, on = column_grouped, how = 'left')
            del grp
            gc.collect()

        elif agragation =='most_common':
            df[column_new] = df[column_origin]
            grp = df.groupby(by = column_grouped)[column_origin].agg(lambda x:x.value_counts().index[0]).reset_index().rename(index = str, columns = {column_new : column_origin})
            df = df.merge(grp, on = column_grouped, how = 'left')
            del grp
            gc.collect()   
        
    return df
#function that returns percentage nulls in a dataframe by feature
def null_ratio(df):
    null_values = df.isna().sum().sum()
    not_null_values = df.count().sum()
    pourcentage =null_values*100/(null_values + not_null_values)
    print("Valeurs nulles: "+ str(null_values))
    print("Valeurs non nulles : "+ str(not_null_values))
    print("Pourcentage des valeurs nulles : "+ str(pourcentage)+ "%")

        
#function that returns columns of a given type        
def column_with_type(df, code):
    columns_type = []
    for c in df.columns :
        if c.startswith(code) :
            columns_type.append(c)
    return columns_type


#getting different types in a dataframe 
def get_dtypes(df):
    for type_ in df.dtypes:
        print (type_)

#correct and convert types in a dataframe
def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(numpy.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(numpy.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(numpy.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df



#get numerical and categorical variables
def get_numerical_categorical(df) :
    numerical = []
    category = []
    for y in df.columns:
        if(df[y].dtype == numpy.float64 or df[y].dtype == numpy.int64 or df[y].dtype == numpy.float32 or df[y].dtype == numpy.int32) :
            numerical.append(y)
        else :
            category.append(y)
            df[y].astype('category')
    return numerical, category

#plot nulls for different features in a dataframe 
def show_null_per_variable  (df) :
    #compter le % de valeur non nulles par année
    df_=df.iloc[:, 1:len(df.columns)+1]
    nulls_per_variable = df_.isna().sum()*100/(df_.isna().sum()+df_.notnull().sum())
    #distribution of the filling rates of the variables with a bar graph
    plt.figure(figsize=(15, 10))
    nulls_per_variable.plot(x ='Variable', y='% not null values', kind = 'bar')
    plt.show()
    del df_
    gc.collect()
    
    
def missing_percentage_variable(df, min_val):
    to_keep=[]
    for i in range(df.shape[1]):
        n_miss = df.iloc[:,i].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        columns= df.columns
        if perc < min_val : 
            print(columns[i]+' > Missing values : %d (%.1f%%)' % ( n_miss, perc))
            to_keep.append(columns[i])
           
    return to_keep
  
    
def remove_missing_columns(train, test, threshold ):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)
    
    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)
    
    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])
    
    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))
    
    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    
    # Drop the missing columns and return
    train = train.drop(columns = missing_columns)
    test = test.drop(columns = missing_columns)
    
    return train, test


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
def transform_flag_variables (df):
    flag_list_values= ['Y','N']
    for c in df.columns :
        unique = df[c].unique()
        if (c.startswith('FLAG')) :
            df[c] = df[c].replace(['Y'], 1.)
            df[c] = df[c].replace(['N'], 0.)
            df[c] = df[c].replace(numpy.nan, -1.)
            df[c] = pd.to_numeric(df[c])
            print(df[c].unique() )   
        elif  (c.startswith('NFLAG')) :
            df[c] = df[c].replace(['Y'], 0.)
            df[c] = df[c].replace(['N'], 1.)
            df[c] = df[c].replace(numpy.nan, -1.)
            df[c] = pd.to_numeric(df[c])
        elif 'CNT' in c :
            #should be positive, replace with 0 negative values
            df.loc[df[c] <0, c] = 0
        elif c ==   'DAYS_BIRTH' :
            #replace with mean
            df.loc[df[c] <0, c] = df[c].mean()
        
    return df      
            
def gini(arr):
    n = arr.size
    coef_ = 2. / n
    const_ = (n + 1.) / n
    weighted_sum = sum([(i+1)*yi for i, yi in enumerate(arr)])
    return coef_*weighted_sum/(arr.sum()) - const_

def lorenz(arr):
    
    scaled_prefix_sum = arr.cumsum() / arr.sum()
    # this prepends the 0 value (because 0% of all people have 0% of all wealth)
    return numpy.insert(scaled_prefix_sum, 0, 0)

#courbe de lorenz pour chacun des indicateurs sélectionnés pour étudier sa concentration : population growth
def plot_lorenz(df, columns_log_transform):
    for c in columns_log_transform :
        arr= df[c].array.to_numpy()
        #arr l'array des valeur de indicator trié
        arr.sort()
        print("Courbe de Lorenz pour l\'indicateur : "+ c)
        print("Indice de gini : "+str(gini(arr)))
        lorenz_curve = lorenz(arr)
        plt.plot(numpy.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve)
        # plot the straight line perfect equality curve
        plt.plot([0,1], [0,1])
        plt.show()
        # show the gini index!
        
    
    
def columns_outliers(df, numerical):
    id_columns = []
    for c in numerical :
        #no outliers on income
        if 'SK_ID' in c or c == 'AMT_INCOME_TOTAL':
            id_columns.append(c)
    treated_columns= [ele for ele in numerical if ele not in id_columns]
    return treated_columns
    
def find_remove_outliers(df, treated_columns):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    #remove anomalies
    df =  df[~((df[treated_columns] < (Q1 - 1.5 * IQR)) |(df[treated_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df 

# ne pas appliquer la transformation log sur les variables avec une valeur dominante = 1 
def transform_variables_log(df, columns):
    for c in columns :
        log_c = 'LOG_'+c
        df[log_c]= df[c]
        for index, row in df.iterrows():
            
            if row[c] == np.nan :
                row[log_c] = np.nan
            else :
                row[log_c] = numpy.log2(abs(float(row[c]))+0.0001)
        df= df.drop(c, axis=1)  
    #df= df.drop(columns, axis=1)
    return df


def KNN_imputer_n(df, n_neighbors, columns):
    X = df.values
    imputer = KNNImputer(n_neighbors=n_neighbors,  weights="uniform", metric='nan_euclidean')
    imputer.fit_transform(X)
    Xtrans = imputer.transform(X)
    df_imputed = pd.DataFrame(Xtrans, columns= columns)
    return df_imputed 


def label_encoder(labelencoder, df, columns):
    for column in columns :
        new_column = column +"_cat"
        df[new_column] = labelencoder.fit_transform(df[column])
    return df

def one_hot_encoder(enc, df, columns):
    for column in columns :
        df_encoded  = pd.get_dummies(df[column],  prefix = column)
        df= pd.concat([df, df_encoded], axis = 1)
        df= df.drop_duplicates()
    df = df.drop(columns, axis = 1)
    return df

def get_label_one_hot_columns (df):
    num, cat = get_numerical_categorical(df)
    label_encoded = []
    one_hot_encoded = []
    for column in cat :
        
        if len(df[column].unique())<3:
            label_encoded.append(column)
        else : 
            one_hot_encoded.append(column)
        return label_encoded, one_hot_encoded

    #normality test : kolmogorov smirnov
def get_non_normal_ditributions_columns (df, list_c): 
    non_norm_cols = []
    for c in df.columns : 
        data = df[c].head(5000).values
        D, p = stats.kstest(data, 'norm')
        #print('Kolmogorov-Smirnov : D : {0} p-value : {1}'.format(D, p))
        if (p==numpy.nan or p <0.5) and c not in list_c : 
            non_norm_cols.append(c)
        del data
        gc.collect()
        
    return non_norm_cols

def convert_status(x):
    if x == 'Closed':
        y = 0
    else:
        y = 1    
    return y

def print_unique_values(df):
    cat_cols = []
    num=[]
    for y in df.columns :
        if (df[y].dtype == numpy.float64 or df[y].dtype == numpy.int64 or df[y].dtype == numpy.float32 or df[y].dtype == numpy.int32) :
            num.append(y)
        else :
            cat_cols.append(y)
            #df[y].astype('category')
    for c in cat_cols: 
        print (df[c].unique())
    return cat_cols    
        
def normalize(x):
    if x<0:
        y = 0
    else:
        y = 1   
    return y

def DPD(DPD):
    
    # DPD is a series of values of SK_DPD for each of the groupby combination 
    # We convert it to a list to get the number of SK_DPD values NOT EQUALS ZERO
    x = DPD.tolist()
    c = 0
    for i,j in enumerate(x):
        if j != 0:
            c += 1
    
    return c 

def count_transaction(min_pay, total_pay):
    
    M = min_pay.tolist()
    T = total_pay.tolist()
    P = len(M)
    c = 0 
    # Find the count of transactions when Payment made is less than Minimum Payment 
    for i in range(len(M)):
        if T[i] < M[i]:
            c += 1  
    return (100*c)/P


def balance_limit(x1, x2):
    
    balance = x1.max()
    limit = x2.max()
    if limit!=0 :
        return (balance/limit)
    else :
        return 0


def update(row, e_dict):
    returned_category=""
    if pd.notnull(row) :
        for key in e_dict:
            if row in e_dict[key]:
                returned_category= key
                break
    return returned_category

def unify_categories_values(df, column, column_dict):
    df[column]= df[column].apply (lambda row :  update(row, column_dict)) 
    return df



def retrieve_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]
            
def unify(df, dict_list):
    for column_dict in dict_list :
        s_col = retrieve_name(column_dict)
        print(s_col )
        column = s_col[0: s_col.index('_dict')]
        print(column)
        df = unify_categories_values(df, column, column_dict)
        
        
def get_subsets_qualitative_column (df, qualitative_column, quantitative_column, grps) : 
    subsets_list = []
    for grp in grps :
        subsets_list.append(df[quantitative_column][df[qualitative_column]==grp])
    return subsets_list
def test_anova(df, qualitative_column, quantitative_column) :
#Create a boxplot
    df.boxplot(quantitative_column, by=qualitative_column, figsize=(12, 8))
    grps = df[qualitative_column].unique()
    subsets_list = get_subsets_qualitative_column (df, qualitative_column, quantitative_column, grps)
    d_data = {grp:df[quantitative_column][df.quantitative_column == grp] for grp in grps}
    k = len(pd.unique(df.qualitative_column))  
    N = len(df.values)  
    del subsets_list
    gc.collec()
    n = df.groupby(qualitative_column).size()[0] 
    print(n)

def one_way_anova(df, qualitative_column, quantitative_column) :
    
    grps = df[qualitative_column].unique()
    list_0 = df[quantitative_column][df[qualitative_column]== 0.]
    list_1 = df[quantitative_column][df[qualitative_column]== 1.]
    
    F, p = stats.f_oneway(list_0, list_1)
    print('F=%.3f, p=%.3f' % (F, p))
    if p > 0.05:
        return False
    else:
        return True

    
def test_anova(df, qualitative_column, quantitative_column) :
#Create a boxplot
    df.boxplot(quantitative_column, by=qualitative_column, figsize=(12, 8))
    grps = df[qualitative_column].unique()
    d_data = {grp:df[quantitative_column][df.quantitative_column == grp] for grp in grps}
    k = len(pd.unique(df.qualitative_column))  
    N = len(df.values)  
    gc.collec()
    n = df.groupby(qualitative_column).size()[0] 
    print(n)
        


def handle_unbalanced_data(X, y):
    sm = SMOTE(random_state=42)
    X_sm, y_sm = sm.fit_resample(np.array(X), y.ravel())

    print(f'''Shape of X before SMOTE: {X.shape} Shape of X after SMOTE: {X_sm.shape}''')
    
    v = pd.DataFrame(y_sm).value_counts(normalize=True) * 100
    print(f'''Balance of positive and negative classes (%):{v}''')
    X= pd.DataFrame(X_sm, columns = X.columns)
    y= pd.DataFrame(y_sm, columns = ['TARGET'])
    return X, y

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (abs(corr_matrix.iloc[i, j]) >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return(dataset)
def create_tune_model(model_name):
    model  = create_model(model_name)
    tuned_model = tune_model(model, n_iter=10, optimize = 'Precision')
    print('Residual Plot')
    plot_model(tuned_model)
    print('Prediction Error')
    plot_model(tuned_model, plot = 'error')
    #print('Feature Importances')
    #plot_model(tuned_model, plot = 'feature')
    print('Model evaluation')
    evaluate_model(tuned_model)
    print('Model interpretation')
    interpret_model(tuned_model)
    

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def Find_Optimal_Threshold(target, predicted):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    #i = np.arange(len(tpr)) 
    #roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    #roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold)
    plot_roc_curve(fpr, tpr)
    return optimal_threshold

def roc_curve_aux (trainX, testX, trainy, testy, model) :

    # generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(testy))]
    # fit a model by model name variable
    model.fit(trainX, trainy)
    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
        # calculate scores
    ns_auc = roc_auc_score(testy, ns_probs)
    lr_auc = roc_auc_score(testy, lr_probs)
        # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
        # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(testy, lr_probs)
        #plot curve
    plot_roc_curve (ns_fpr, ns_tpr, lr_fpr, lr_tpr)
    