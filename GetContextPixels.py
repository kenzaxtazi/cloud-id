import DFAnalysis as DFA
import DataPreparation as DP


df = DFA.df_model_agreement('/home/hep/trz15/Matched_Pixels2/Calipso/P4', MaxDist=1000000, MaxTime=1000000, model='Net1_FFN', model_network='Network1')

bad = DFA.get_bad_classifications(df)

context_df = DFA.get_contextual_dataframe(bad)

context_df.to_pickle('ContextualPixels.pkl')
