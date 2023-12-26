import pandas as pd
import statsmodels.api as sm
from typing import List


class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            forward_list: (python list) contains significant features. Each feature
            name is a string
        """
        fwd = []
        cols = data.columns.tolist() 

        while cols: 
            pvals =[]
            for c in cols: 
                p_value = (sm.OLS(target, sm.add_constant(data[fwd + [c]])).fit().pvalues[c])
                pvals.append(p_value)


            minp = min(pvals)
            if minp > significance_level: 
                break 

            index = pvals.index(minp)
            fwd.append(cols[index])
            cols.pop(index)
        
        return fwd 


    @staticmethod
    def backward_elimination(
        data: pd.DataFrame, target: pd.Series, significance_level: float = 0.05
    ) -> List[str]:  # 9 pts
        """
        Args:
            data: (pandas data frame) contains the feature matrix
            target: (pandas series) represents target feature to search to generate significant features
            significance_level: (float) threshold to reject the null hypothesis
        Return:
            backward_list: (python list) contains significant features. Each feature
            name is a string
        """

        #col_rm =[] #list of cols to remove during back 
        cols = data.columns.tolist()

        while cols: 
            p_values=(sm.OLS(target,sm.add_constant(data[cols])).fit().pvalues[cols].tolist())
            max_p=max(p_values)
            if max_p<significance_level:
                break
            index=p_values.index(max_p)
            cols.pop(index)
        return cols





