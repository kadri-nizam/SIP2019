def plot(array):
    if(array):
        for i in range(len(fluxesDE)):
            sns.set_style('darkgrid')                     #plot DE
            g = sns.lineplot(correct,fluxesDE[i], color = 'dodgerblue')  
            g.set(xlabel='Lambda', ylabel='Flux', title='DE Spectra')
            plt.show()
        for i in range(len(fluxesUDG)):           #plot UDG
            g = sns.lineplot(correct,fluxesUDG[i], color = 'orchid')   
            g.set(xlabel='Lambda', ylabel='Flux', title='UDG Spectra')
            plt.show()
        for i in range(len(fluxesVCC)):           #plot UDG
            g = sns.lineplot(correct,fluxesVCC[i], color = 'mediumturquoise')   
            g.set(xlabel='Lambda', ylabel='Flux', title='VCC Spectra')
            plt.show()
    else:
        wow = fluxesDE.T
        wow['Correct'] = correct 
        for i in range(len(wow.columns)-1):
            g = wow.plot(kind='line',x='Correct',y=i,color='dodgerblue')
            g.set(xlabel='Lambda', ylabel='Flux', title='DE Spectra')
            plt.show()
        wow = fluxesUDG.T
        wow['Correct'] = correct 
        for i in range(len(wow.columns)-1):
            g = wow.plot(kind='line',x='Correct',y=i,color='orchid')
            g.set(xlabel='Lambda', ylabel='Flux', title='UDG Spectra')
            plt.show()
        wow = fluxesVCC.T
        wow['Correct'] = correct 
        for i in range(len(wow.columns)-1):
            g = wow.plot(kind='line',x='Correct',y=i,color='mediumturquoise')
            g.set(xlabel='Lambda', ylabel='Flux', title='VCC Spectra')
            plt.show()     
