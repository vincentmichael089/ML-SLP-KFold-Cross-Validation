df_k1.at[0,'target']= df_k1.at[0,'x1']*df_k1.at[0,'t1']+df_k1.at[0,'x2']*df_k1.at[0,'t2']+df_k1.at[0,'x3']*df_k1.at[0,'t3']+df_k1.at[0,'x4']*df_k1.at[0,'t4']+df_k1.at[0,'bias']
df_k1.at[0,'sigmoid']= 1/(1+math.exp(-df_k1.at[0,'target']))
if df_k1.at[0,'sigmoid']>0.5:
    df_k1.at[0,'prediction']= 1
else:
    df_k1.at[0,'prediction']= 0   
df_k1.at[0,'error']= math.pow(df_k1.at[0,'binary']-df_k1.at[0,'sigmoid'],2)
df_k1.at[0,'dt1']= 2*df_k1.at[0,'x1']*(df_k1.at[0,'binary']-df_k1.at[0,'sigmoid'])*(1-df_k1.at[0,'sigmoid'])*df_k1.at[0,'sigmoid']
df_k1.at[0,'dt2']= 2*df_k1.at[0,'x2']*(df_k1.at[0,'binary']-df_k1.at[0,'sigmoid'])*(1-df_k1.at[0,'sigmoid'])*df_k1.at[0,'sigmoid']
df_k1.at[0,'dt3']= 2*df_k1.at[0,'x3']*(df_k1.at[0,'binary']-df_k1.at[0,'sigmoid'])*(1-df_k1.at[0,'sigmoid'])*df_k1.at[0,'sigmoid']
df_k1.at[0,'dt4']= 2*df_k1.at[0,'x4']*(df_k1.at[0,'binary']-df_k1.at[0,'sigmoid'])*(1-df_k1.at[0,'sigmoid'])*df_k1.at[0,'sigmoid']
df_k1.at[0,'dbias']= 2*(df_k1.at[0,'binary']-df_k1.at[0,'sigmoid'])*(1-df_k1.at[0,'sigmoid'])*df_k1.at[0,'sigmoid']
