# This code implements the method in the following paper for imputing
# missing weight measurement at time T for patients who dropped out before
# time T.

"""
 References:
 1. Rotnitzky, Andrea, and James M. Robins. "Semiparametric regression estimation 
 in the presence of dependent censoring." Biometrika (1995): 805-820.
 2. Marie Davidian's course notes: http://www4.stat.ncsu.edu/~davidian/st790/notes/chap5.pdf
    page 143
"""

# Author: Wei Jiang
# April 20, 2017

import numpy as np
import pandas as pd
from scipy.optimize import fmin_bfgs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, classification_report
# from read_data import get_final_data
from RadOnc import dummy_categorical
import matplotlib.pyplot as plt
import itertools
from copy import deepcopy


class imputer():

    def _prepare_data(self, filename, min_date=0, max_date=360, sampling_freq=7):
        """
        filename: str, filename for the csv file which stores both baseline covariates and 
                    temporal prediction target data.
                    e.g., "df_final.csv"
        """
        df = pd.read_csv(filename)
        df = df.ix[:,1:]

        dates_wanted = np.arange(min_date, max_date, step = sampling_freq)

        weight_col_names = ['Day_' + str(i) for i in dates_wanted]
        # print weight_col_names

        # obtain indicator matrix for weight
        # 1 means observed
        indicator_observed = np.isfinite(df[weight_col_names])
        
        indicator_df = pd.DataFrame(indicator_observed, columns = weight_col_names)
        indicator_df['patientID'] = df['patientID'].values
        indicator_df = indicator_df.set_index('patientID')

        # indicator_df.to_csv("indicator.csv")

        # build logistic regression model for missingness from day 1 to last day

        cat = ['tStage','nStage','mStage','overallStage','diagnosis','Chemo_completed','race_combined','referenceYear','attending',
              'technique','gender','p16_ever_positive','hpv_ever_positive','comb_hpv','peg_ever_used',
              'ng_ever_used']
        df = dummy_categorical(df, cat)
        return df, indicator_df, weight_col_names

    def _get_complete_case_patientID(self, indicator_df, day):
        '''
        params:
            indicator_df: dataframe, output from function self._prepare_data()
            day: string, e.g., 'Day_102'
        return: dataframe
        '''
        observed_patID = indicator_df.loc[indicator_df[day].values,:].reset_index()[['patientID']]
        return observed_patID



    def _remove_days(self, day_to_predict, all_days):
        '''
        given a day to predict, return the column names of days after as a list of string
        does not contain day_to_predict itself
        param:
            day_to_predict: string, e.g., 'Day_-3.0'
            all_days: list of string
        '''
        day = float(day_to_predict.split('_')[1])
        days = np.array([float(day_i.split('_')[1]) for day_i in all_days])

        to_remove_mask = days > day
        days_to_remove = list(itertools.compress(all_days, to_remove_mask))

        return days_to_remove

    def _get_observed_df(self, day_i, all_days, df, indicator_df, sampling_freq = 7):
        '''
        get the subset of patients data that are still observed at time day_i-1
        '''
        day_i_before = 'Day_' + str(int(day_i.split('_')[1])-sampling_freq)
        
        observed_df = df.loc[indicator_df[day_i_before].values,:]
        days_to_remove = self._remove_days(day_i, all_days )
        observed_df = observed_df.drop(days_to_remove, axis = 1)

        return observed_df


    def test_get_observed_df(self, df, all_days, indicator_observed,sampling_freq = 7):
        df = self._get_observed_df('Day_102.0', all_days, df, indicator_observed,sampling_freq = 7)
        df.to_csv('observed_Day102.csv')

    def test_remove_days(self):
        day = 'Day_-3.0'
        print self._remove_days(day, weight_col_names)

    def _get_auc(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def _build_logit_model(self, df, indicator_observed, weight_col_names):
        
        predict_score_all_days = []
        # store training AUC for each logistic regression model
        training_auc = []

        for day in weight_col_names[1:]:
            lg = LogisticRegression(C =10000000000, solver='lbfgs',
             max_iter = 1000000000)
            # days_to_remove = self._remove_days(day, weight_col_names)
            # lg = RandomForestClassifier(n_estimators = 100)

            df_day_i = self._get_observed_df(day, weight_col_names, df, indicator_observed,sampling_freq = 7)
            # print df_day_i['patientID']
            # patientID column has not been removed yet
            X_df = df_day_i.drop(['patientID',day], axis = 1)
            X_df = X_df.T.drop_duplicates().T
            X = X_df.values
            # Check if the weight is missing at day i
            y = np.isfinite(df_day_i[day].values).astype(int)

            ## if y has only one unique value, most likely, all 1, which means no patient has weight missing at day i 
            ## for the patients who are still present at day i-1, then we can't train the data using supervised learning algorithms
            ## as there is only one class. I set all the probabilities to be 1 in this case.

            if len(set(y)) == 2:
                
                lg.fit(X, y)
                # formular = day + " ~" + "+".join(df_day_i.columns - [day, 'patientID'])
                # print formular
                # model = smf.glm(formular = formular, data = df_day_i, family = sm.families.Binomial()).fit()
                # print model.summary()
                predictions = lg.predict_proba(X)[:,1]
                pred_auc = self._get_auc(y, predictions)
            elif len(set(y)) == 1:
                predictions = np.ones((X.shape[0],1)).flatten()
                pred_auc = np.nan

            predict_score = pd.DataFrame({'patientID': df_day_i['patientID'].values, day: predictions })
            predict_score_all_days.append(predict_score)

            training_auc.append(pred_auc)
            # print predict_score
            # predict_score.to_csv('predicted_score.csv')
        predicted_score_df = reduce(lambda x, y: pd.merge(x,y, on = 'patientID', how = 'outer'), predict_score_all_days)
        predicted_score_df = predicted_score_df.fillna(value = 1)
        # predicted_score_df.loc[~indicator_observed['Day_109.0'].values,:]
        return predicted_score_df, training_auc

    def get_inverse_probs(self, day = 'Day_109.0'):
        '''
        Get the inverse probability weights 1/pi_i for the complete cases
        params:
            day: string
        return: 
            complete_case_score: dataframe
        '''
        df, indicator_observed, weight_col_names = self._prepare_data()
        
        prob_df, AUCs = self._build_logit_model(df, indicator_observed, weight_col_names)
        
        
        propensity_score = np.prod(prob_df.drop('patientID',axis=1).values, axis=1)
        propensity_score_df = pd.DataFrame({'patientID':prob_df['patientID'].values, 'observing_score': propensity_score})
        
        complete_case_patID = self._get_complete_case_patientID(indicator_df, day)
        complete_case_score = pd.merge(complete_case_patID, propensity_score_df, how='inner', on='patientID')
        complete_case_score['weight'] = 1/complete_case_score['observing_score']
        return complete_case_score



    # def solve_inverse_prob_estimate_equation()
    def estimate_weight_likelihood(predicted_score_df, indicator_observed):
        indicator_observed = indicator_observed.astype(int)
        print indicator_observed.head()
        lik_ls =[]
        pt_ID = predicted_score_df['patientID'].values
        predicted_score_df = predicted_score_df.set_index('patientID')
        for i in pt_ID:
            prob = predicted_score_df.loc[i,:].values
            print prob
            observeness = indicator_observed.loc[i,:].values[1:]
            print observeness
            lik = 1.0
            for p, o in zip(prob, observeness):

                lik = lik*np.power(p, o)*np.power(1-p, 1-o)
                if o==0:
                    break
            print lik
            lik_ls.append(lik)
        likelihood_df = pd.DataFrame({'patientID': predicted_score_df.index, 'likelihood':lik_ls})
        return likelihood_df


    def impute_main(self, filename1, day_to_impute = None):
        """
        filename1: string, e.g., "trismus_final.csv", dataframe with prediction target (toxicity) preprocessed (interpolated, outlier
        removed) then merged with baseline covariates. "trismus_final.csv" was generated by running Trismus_preprocess.ipynb.
        day_to_impute: str, e.g., 'Day_147', this is the date for which we want to impute missing data for.
                        if None, then impute missing data for all the dates used for computing missing data.
        """
        df, indicator_observed, weight_col_names = self._prepare_data(filename1)
        min_date = weight_col_names[0]
        
        # First, fit longitudinal logistic regression to compute the inverse  probability
        # i.e., fit a separate logistic regression for each time t using patients who didn't 
        # drop out at time t-1

        prob_df, AUCs = self._build_logit_model(df, indicator_observed, weight_col_names)

        imputed_df_ls = []
        imputed_df_ls.append(df[['patientID',weight_col_names[0]]])

        if day_to_impute not in weight_col_names[1:]:
            raise Exception("Please pick another date to impute, date dosn't match with dates used for imputation process!")

        if day_to_impute:
            dates_to_impute_ls = list(day_to_impute)
        else:
            dates_to_impute_ls = weight_col_names[1:]
        for day_i in dates_to_impute_ls:
            propensity_score = np.prod(prob_df.drop('patientID',axis=1).values, axis=1)
            propensity_score_df = pd.DataFrame({'patientID':prob_df['patientID'].values, 'observing_score': propensity_score})

            complete_case_patID = self._get_complete_case_patientID(indicator_observed, day_i)
            complete_case_score = pd.merge(complete_case_patID, propensity_score_df, how='inner', on='patientID')

            complete_case_score['weight'] = 1/complete_case_score['observing_score']
            
            # Get the independent features for regression, chose 'Day_-10.0' weight as baseline weight
            tStage_names = [i  for i in df.columns if 'tStage' in i]
            nStage_names = [i for i in df.columns if 'nStage' in i]
            regress_features = df[['patientID', 'ageAtRefDate', ' D90',min_date] + tStage_names + nStage_names]
            # Add a constant column for modelling intercept for linear regression
            regress_features['intercept'] = 1
            regress_features_with_weights = pd.merge(regress_features, complete_case_score[['patientID','weight']], how='inner', on='patientID')
            regress_features_with_weights = regress_features_with_weights.set_index('patientID')
            # drop duplicate columns (should be all 0 columns) due to dummirize categorical features and getting the complete case
            regress_features_with_weights = regress_features_with_weights.T.drop_duplicates().T 

            regress_features_weighted_complete_case = regress_features_with_weights.drop('weight', axis='columns').multiply(regress_features_with_weights['weight'], axis='rows')
            # Get the corresponding dependent variable Y
            Y = df[['patientID', day_i]]
            Y = Y.set_index('patientID')
            final_data_df = pd.merge(regress_features_weighted_complete_case, Y, how='inner', left_index=True, right_index=True )
            ## Find all zero columns
            zero_columns = final_data_df.columns[(final_data_df == 0).all()]
            final_data_df = final_data_df.drop(zero_columns, axis = 'columns')

            D_matrix = final_data_df.drop(day_i, axis='columns').values.T
            Y = final_data_df[day_i].values
            X = final_data_df.divide(final_data_df['intercept'], axis='rows').drop(day_i, axis='columns').values

            # Solve for system of equations for regression coefficients
            a = np.dot(D_matrix, X)
            b = np.dot(D_matrix, Y)
            betas = np.linalg.solve(a, b)

            betas_df = pd.DataFrame({'variables':final_data_df.drop(day_i,axis='columns').columns, 'betas': betas})
            betas_df = betas_df.sort_values(by='betas')
            to_impute_df = regress_features.set_index('patientID')[betas_df['variables']]

            to_impute_df = to_impute_df.drop(to_impute_df.index[np.where(to_impute_df.isnull().values == True)[0]])
            predicted_weight = np.dot(to_impute_df.values, betas_df['betas'].values)
            predicted_weight_df = pd.DataFrame({'patientID':to_impute_df.index, day_i: predicted_weight})

            ## Mix the predicted weights with the complete case weights
            predicted_weight_df_mix = deepcopy(predicted_weight_df)
            predicted_weight_df_mix = predicted_weight_df_mix.set_index('patientID')
            predicted_weight_df_mix.loc[complete_case_patID['patientID'], day_i] = final_data_df.loc[complete_case_patID['patientID'],day_i]

            imputed_df_ls.append(predicted_weight_df_mix.reset_index())

        imputed_df = reduce(lambda left, right: pd.merge(left, right, on = 'patientID', how='outer'), imputed_df_ls)

        return imputed_df

    def test_estimate_weight_likelihood():
        df, indicator_observed, weight_col_names = self._prepare_data()
        # self.test_get_observed_df(df, weight_col_names, indicator_observed)
        prob_df = self._build_logit_model(df, indicator_observed, weight_col_names)
        likelihood_df = estimate_weight_likelihood(prob_df, indicator_observed)
        df = pd.merge(likelihood_df, indicator_observed[['Day_109.0']].reset_index(), on="patientID")
        df.to_csv('likelihood.csv')

    def test_plot_roc_for_logit_fits_as_whole():
        '''
        This fitness test may not make sense, needs further check
        '''

        df, indicator_observed, weight_col_names = self._prepare_data()
        
        prob_df, AUCs = self._build_logit_model(df, indicator_observed, weight_col_names)
        propensity_score = np.prod(prob_df.drop('patientID',axis=1).values, axis=1)
        propensity_score_df = pd.DataFrame({'patientID':prob_df['patientID'].values, 'observing_score': propensity_score})
        

        check_lg_fit_df = pd.merge(indicator_observed[['Day_109.0']].reset_index(), propensity_score_df, how = 'inner', on='patientID')
        y_true = check_lg_fit_df['Day_109.0'].values.astype(int)
        y_pred = check_lg_fit_df['observing_score'].values
        temp = deepcopy(y_pred)
        temp[y_pred<=0.5] = 0
        temp[y_pred>0.5] = 1
        # y_pred = estimate_weight_likelihood(prob_df, indicator_observed)
        print(classification_report(y_true, temp))
        print 'Y true:'
        print y_true
        # compute auc, draw roc curve

        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure()
       
        plt.plot(fpr, tpr, color='darkorange',
                  label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_observing_prob.png')
        plt.show()

if __name__ == '__main__':
    im = imputer()
    df, indicator_observed, weight_col_names = im._prepare_data()
    prob_df, AUCs = im._build_logit_model(df, indicator_observed, weight_col_names)
    

    

    



