'''
Päivitetty 16.11.2025
Aki Taanila aki@taanila.fi
'''

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, pearsonr

df = pd.DataFrame()
kategoriset = []
dikotomiset = []
kvantit = []
listat = {}
tunnusluvut = ['Lukumäärä', 'Keskiarvo', 'Keskihajonta', 'Pienin', 
               'Alaneljännes', 'Mediaani', 'Yläneljännes', 'Suurin']


def arvot():
    '''Kategoriset- ja dikotomiset-listojen muuttujien ainutkertaiset arvot.'''

    for muuttuja in df[kategoriset+dikotomiset]:
        try:
            print(muuttuja, np.unique(df[muuttuja]))
        except:
            print(muuttuja, pd.unique(df[muuttuja]))
    

def frekv(muuttuja):
    '''Frekvenssitaulukko muuttujalle.'''
    
    df1 = pd.crosstab(df[muuttuja], columns='f')
    df1.columns.name = ''
    n = df1['f'].sum()
    df1['%'] = df1['f']/n * 100

    # muuttujan tekstimuotoiset arvot
    if muuttuja in listat.keys():
        if len(listat[muuttuja]) == len(df1.index):
            df1.index = listat[muuttuja]

    df1.loc['Yhteensä'] = df1.sum()

    return df1.style.format({'f':'{:.0f}', '%':'{:.1f} %'})


def dikot():
    '''Laskee dikotomiset-listan muuttujien yhteenvedon.'''
    
    if len(dikotomiset) > 1:
        df1 = df[dikotomiset].sum().to_frame('f').sort_values('f', ascending=False)
        n = df.shape[0]
        df1['% vastaajista'] = df1['f']/n * 100
        print(f'n = {n}')

        return df1.style.format({'f':'{:.0f}', '% vastaajista':'{:.1f} %'})


def tunnu():
    '''Laskee kvantit-listan muuttujien tilastolliset tunnusluvut.'''

    if kvantit:
        if len(kvantit) == 1:
            df1 = df[kvantit].describe().to_frame()
        else:
            df1 = df[kvantit].describe()
        df1.index = tunnusluvut

        return df1


def risti(muuttuja1, muuttuja2):
    '''Ristiintaulukointi sarakeprosentteina muuttujille muuttuja1 ja muuttuja2.'''

    df1 = pd.crosstab(df[muuttuja1], df[muuttuja2], margins=True, 
                      normalize='columns') * 100
    dfn = pd.crosstab(df[muuttuja1], df[muuttuja2], margins=True)
    
    # muuttujien tekstimuotoiset arvot
    if muuttuja1 in listat.keys():
        if len(listat[muuttuja1]) == len(df1.index):
            df1.index = listat[muuttuja1]
    if muuttuja2 in listat.keys():
        if len(listat[muuttuja2]) == len(df1.columns) - 1:
            df1.columns = listat[muuttuja2] + ['Yhteensä']
        else:
            df1.rename(columns={'All':'Yhteensä'}, inplace=True)
    else:
            df1.rename(columns={'All':'Yhteensä'}, inplace=True)

    # n-arvot
    n_arvot = {}
    for i in range(dfn.shape[1]):
        n_arvot.update({str(df1.columns[i]): int(dfn.iloc[:, i].sum()/2)})
    print(f'n-arvot: {n_arvot}')

    return df1.style.format('{:.1f} %')


def risti_lkm(muuttuja1, muuttuja2):
    '''Ristiintaulukointi lukumäärinä muuttujille muuttuja1 ja muuttuja2.'''

    df1 = pd.crosstab(df[muuttuja1], df[muuttuja2], margins=True)
    
    # muuttujien tekstimuotoiset arvot
    if muuttuja1 in listat.keys():
        if len(listat[muuttuja1]) == len(df1.index) - 1:
            df1.index = listat[muuttuja1] + ['Yhteensä']
        else:
            df1.rename(index={'All':'Yhteensä'}, inplace=True)
    if muuttuja2 in listat.keys():
        if len(listat[muuttuja2]) == len(df1.columns) - 1:
            df1.columns = listat[muuttuja2] + ['Yhteensä']
        else:
            df1.rename(columns={'All':'Yhteensä'}, inplace=True)
        
    return df1


def risti_khi(muuttuja1, muuttuja2):
    '''Ristiintaulukointi ja khiin neliö -testi muuttujille muuttuja1 ja muuttuja2.'''  
    
    khi, p, dof, expected = chi2_contingency(pd.crosstab(df[muuttuja1], df[muuttuja2]))
    print(
          f'Khiin neliö = {khi:.2f}, p-arvo = {p:.3f}, '
          f'vapausasteet = {dof},')
    print(
          f'alle viiden suuruisia odotettuja frekvenssejä '
          f'{((expected<5).sum()/expected.size*100).round(2)} %.')
    print('\nOdotetut frekvenssit:')

    # muuttujien tekstimuotoiset arvot
    df1 = pd.DataFrame(expected)
    if muuttuja1 in listat.keys():
        if len(listat[muuttuja1]) == len(df1.index):
            df1.index = listat[muuttuja1]
    if muuttuja2 in listat.keys():
        if len(listat[muuttuja2]) == len(df1.columns):
            df1.columns = listat[muuttuja2]
    
    return df1


def risti_dikot(muuttuja):
    '''Ristiintaulukointi muuttujan ja dikotomiset-listan muuttujien välille.'''

    if len(dikotomiset) > 1:
        df1 = df.groupby(muuttuja, observed=True)[dikotomiset].sum()
        dfn = pd.crosstab(df[muuttuja], 'n')
        
        # Kategorisen muuttujan tekstimutoiset arvot
        if muuttuja in listat.keys():
            if len(listat[muuttuja]) == len(df1.index):
                df1.index = listat[muuttuja]
                dfn.index = listat[muuttuja]

        # Prosentit ja n-arvot
        n_arvot = {}
        for i in range(df1.shape[0]):
            df1.iloc[i, :] = df1.iloc[i, :]/dfn.iloc[i, 0] * 100
            n_arvot.update({str(df1.index[i]): int(dfn.iloc[i, 0])})
        print(f'n-arvot: {n_arvot}')

        return df1.T.style.format('{:.1f} %')


def risti_tunnu(muuttuja1, muuttuja2):
    '''
    Muuttujan muuttuja2 tunnuslukuja kategorisen muuttujan muuttuja1 määräämissä 
    ryhmissä. Jos kategorinen muuttuja määrittää täsmälleen kaksi ryhmää, niin 
    funktio palauttaa myös kahden riippumattoman otoksen t-testin tuloksen.
    '''

    df1 = df.groupby(muuttuja1, observed=True)[muuttuja2].describe().T
    df1.index = tunnusluvut

    if muuttuja1 in listat.keys():
        if len(listat[muuttuja1]) == len(df1.columns):
            df1.columns = listat[muuttuja1]

    kategoriat = df[muuttuja1].dropna().unique()
    if len(kategoriat) == 2:
        ryhma1 = df[muuttuja2][df[muuttuja1]==kategoriat[0]]
        ryhma2 = df[muuttuja2][df[muuttuja1]==kategoriat[1]]
        t, p = ttest_ind(ryhma1, ryhma2, equal_var=False, nan_policy='omit')
        print(f't = {t:.2f}, p-arvo = {p:.3f}.')

    return df1


def korre():
    '''Laskee kvantit-listan muuttujien väliset Pearsonin korrelaatiokertoimet.'''
    
    if kvantit:
        dfr = df[kvantit].corr(min_periods=10, numeric_only=True)
        
        return dfr


def tehosta_solut(arvo):
    '''Tehostaa solut, joiden arvo on pienempi kuin 0.05.'''
    
    if arvo < .05:
        return 'background-color: pink'
    else:
        return ''


def korre_p():
    '''Laskee kvantit-listan muuttujien väliset Pearsonin korrelaatiokertoimien p-arvot.'''
    
    if kvantit:
        dfp = pd.DataFrame(index=df[kvantit].columns, 
                           columns=df[kvantit].columns).astype('float')
        for muuttuja1 in df[kvantit]:
            for muuttuja2 in df[kvantit]:
                if muuttuja1 != muuttuja2:
                    df_dropna = df.dropna(subset=[muuttuja1, muuttuja2])
                    r, p = pearsonr(df_dropna[muuttuja1], df_dropna[muuttuja2])
                    dfp.loc[muuttuja1, muuttuja2] = p
        styled_dfp = dfp.style.map(tehosta_solut)
       
        return styled_dfp
        

def korre_n():
    '''Laskee kvantit-listan muuttujien väliset Pearsonin korrelaatiokertoimien n-arvot.'''

    if kvantit:
        dfn = pd.DataFrame(index=df[kvantit].columns, columns=df[kvantit].columns)
        for muuttuja1 in df[kvantit]:
            for muuttuja2 in df[kvantit]:
                if muuttuja1 != muuttuja2:
                    df_dropna = df.dropna(subset=[muuttuja1, muuttuja2])
                    n = df_dropna.shape[0]
                    dfn.loc[muuttuja1, muuttuja2] = n
    
        return dfn


def frekv_excel(muuttuja):
    '''Frekvenssitaulukko muuttujalle.'''
    
    df1 = pd.crosstab(df[muuttuja], columns='f')
    df1.columns.name = ''
    n = df1['f'].sum()
    df1['%'] = df1['f']/n

    # muuttujan tekstimuotoiset arvot
    if muuttuja in listat.keys():
        if len(listat[muuttuja]) == len(df1.index):
            df1.index = listat[muuttuja]

    df1.loc['Yhteensä'] = df1.sum()

    return df1


def risti_excel(muuttuja1, muuttuja2):
    '''
    Ristiintaulukointi muuttujille muuttuja1 ja muuttuja2.
    Funktio palauttaa ristiintaulukoinnin lisäksi n-arvot ja 
    khiin neliö -testin tuloksen.
    '''
        
    khi_testi = chi2_contingency(pd.crosstab(df[muuttuja1], df[muuttuja2]))
    df1 = pd.crosstab(df[muuttuja1], df[muuttuja2], margins=True)
    df2 = pd.crosstab(df[muuttuja1], df[muuttuja2], margins=True, 
                      normalize='columns')

    # muuttujien tekstimuotoiset arvot
    if muuttuja1 in listat.keys():
        if len(listat[muuttuja1]) == len(df1.index) - 1:
            df1.index = listat[muuttuja1] + ['Yhteensä']
            df2.index = listat[muuttuja1]
        else:
            df1.rename(index={'All':'Yhteensä'}, inplace=True)
    else:
        df1.rename(index={'All':'Yhteensä'}, inplace=True) 
    if muuttuja2 in listat.keys():
        if len(listat[muuttuja2]) == len(df1.columns) - 1:
            df1.columns = listat[muuttuja2] + ['Yhteensä']
            df2.columns = listat[muuttuja2] + ['Yhteensä']
        else:
            df1.rename(columns={'All':'Yhteensä'}, inplace=True)
            df2.rename(columns={'All':'Yhteensä'}, inplace=True)
    else:
        df1.rename(columns={'All':'Yhteensä'}, inplace=True)
        df2.rename(columns={'All':'Yhteensä'}, inplace=True)

    n_arvot = ['n']
    for i in range(df1.shape[1]):
        n_arvot.append(df1.iloc[-1, i])

    return df2, n_arvot, khi_testi


def dikot_excel():
    '''Yhteenveto dikotomisille muuttujille.'''

    df1 = df[dikotomiset].sum().to_frame('f').sort_values('f', ascending=False)
    n = df.shape[0]
    df1['% vastaajista'] = df1['f']/n
    
    return df1, n


def risti_dikot_excel(muuttuja):
    '''Ristiintaulukointi muuttujan ja dikotomisten muuttujien välille.'''

    df1 = df.groupby(muuttuja)[dikotomiset].sum()
    dfn = pd.crosstab(df[muuttuja], 'n')   # n-arvot
    dfn.columns.name = ''

    # Prosenttien laskenta
    n_arvot = ['n']
    for i in range(df1.shape[0]):
        df1.iloc[i, :] = df1.iloc[i, :]/dfn.iloc[i, 0]
        n_arvot.append(dfn.iloc[i, 0])

    # kategorisen muuttujan tekstimutoiset arvot
    if muuttuja in listat.keys():
        if len(listat[muuttuja]) == len(df1.index):
            df1.index = listat[muuttuja]
            dfn.index = listat[muuttuja]

    return df1.T, n_arvot


def risti_tunnu_excel(muuttuja1, muuttuja2):
    '''
    Muuttujan muuttuja2 tunnuslukuja kategorisen muuttujan muuttuja1 määräämissä 
    ryhmissä. Jos kategorinen muuttuja määrittää täsmälleen kaksi ryhmää, niin 
    funktio palauttaa myös kahden riippumattoman otoksen t-testin tuloksen.
    '''
    
    df1 = df.groupby(muuttuja1, observed=True)[muuttuja2].describe().T
    df1.index = tunnusluvut

    if muuttuja1 in listat.keys():
        if len(listat[muuttuja1]) == len(df1.columns):
            df1.columns = listat[muuttuja1]

    kategoriat = df[muuttuja1].dropna().unique()
    if len(kategoriat) == 2:
        ryhma1 = df[muuttuja2][df[muuttuja1]==kategoriat[0]]
        ryhma2 = df[muuttuja2][df[muuttuja1]==kategoriat[1]]
        testi = ttest_ind(ryhma1, ryhma2, equal_var=False, nan_policy='omit')
    else:
        testi=False

    return df1, testi

 
def korre_excel():
    '''
    Laskee kvantit-listan sisältämien muuttujien väliset Pearsonin korrelaatiokertoimet 
    sekä kertoimiin liittyvät p-arvot ja n-arvot.
    '''
   
    dfr = pd.DataFrame(index=df[kvantit].columns, columns=df[kvantit].columns).astype('float')
    dfp = pd.DataFrame(index=df[kvantit].columns, columns=df[kvantit].columns).astype('float')
    dfn = pd.DataFrame(index=df[kvantit].columns, columns=df[kvantit].columns)
    for muuttuja1 in df[kvantit]:
        for muuttuja2 in df[kvantit]:
            if muuttuja1 != muuttuja2:
                df_dropna = df.dropna(subset=[muuttuja1, muuttuja2])
                r, p = pearsonr(df_dropna[muuttuja1], df_dropna[muuttuja2])
                n = df_dropna.shape[0]
                dfr.loc[muuttuja1, muuttuja2] = r
                dfp.loc[muuttuja1, muuttuja2] = p
                dfn.loc[muuttuja1, muuttuja2] = n
    return dfr, dfp, dfn


def ana_excel(tiedosto):
    '''
    Frekvenssitaulukot, dikotomisten muuttujien yhteenvedot, 
    tilastolliset tunnusluvut, ristiintaulukoinnit ja korrelaatiot Excel-tiedostoon.
    '''

    # Excel-tiedoston luonti
    writer = pd.ExcelWriter(tiedosto, engine='xlsxwriter')
    workbook = writer.book

    # Exceliin tehtävät muotoilut
    decimal3_format = workbook.add_format({'num_format':'0.000'})
    percent_format = workbook.add_format({'num_format':'0.0 %'})
    header_format = workbook.add_format({'align': 'right', 'bottom': 1})
    title_format = workbook.add_format({'align': 'left', 'bottom':1})
    index_format = workbook.add_format({'align': 'left', 'border':0})
    bold_format = workbook.add_format({'bold': True})
    right_format = workbook.add_format({'align':'right'})

    # Frekvenssitaulukot
    if kategoriset:
        rivi = 2    # Excelin rivinumero, johon seuraava taulukko kirjoitetaan
        ws1 = workbook.add_worksheet('frekvenssitaulukot')
        for muuttuja in kategoriset:
            df1 = frekv_excel(muuttuja)
            df1.to_excel(writer, sheet_name='frekvenssitaulukot', startrow=rivi)
            ws1.write(rivi, 0, muuttuja, title_format)
            for sarake, arvo in enumerate(df1.columns.values):
                ws1.write(rivi, sarake+1, arvo, header_format)
            for rivinumero, arvo in enumerate(df1.index.values):
                ws1.write(rivi+rivinumero+1, 0, arvo, index_format)
            rivi = rivi + df1.shape[0] + 2
        ws1.set_column('C:C', cell_format=percent_format)
        ws1.write(0, 0, 'Kategoristen muuttujien frekvenssitaulukot', bold_format)
    
    # Ristiintaulukoinnit
    if len(kategoriset) > 1:
        rivi = 3 # Excelin rivinumero, johon seuraava taulukko kirjoitetaan
        ws2 = workbook.add_worksheet('ristiintaulukoinnit')
        for muuttuja1 in kategoriset:
            for muuttuja2 in kategoriset:
                if muuttuja1 != muuttuja2:
                    df2, n_arvot, khi_testi = risti_excel(muuttuja1, muuttuja2)
                    df2.to_excel(writer, sheet_name='ristiintaulukoinnit', startrow=rivi)
                    for sarake, n in enumerate(n_arvot):
                        ws2.write(rivi+df2.shape[0]+1, sarake, n)
                    ws2.write(rivi, 0, muuttuja1, title_format)
                    ws2.write(rivi-1, 1, muuttuja2)
                    for sarake, arvo in enumerate(df2.columns):
                        ws2.write(rivi, sarake + 1, arvo, header_format)
                    for rivinumero, arvo in enumerate(df2.index):
                        ws2.write(rivi+rivinumero+1 , 0, arvo, index_format)
                    for rivinumero in range(rivi+1, rivi+df2.shape[0]+1):
                        ws2.set_row(rivinumero, cell_format=percent_format)
                    ws2.write(rivi, df2.shape[1]+2, 
                              f'Khiin neliö = {khi_testi[0]:.3f}, ' 
                              f'p-arvo = {khi_testi[1]:.3f}, vapausasteet = {khi_testi[2]}.')
                    ws2.write(rivi+1, df2.shape[1]+2, 
                              f'Alle viiden suuruisia odotettuja frekvenssejä '
                              f'{((khi_testi[3]<5).sum()/khi_testi[3].size*100).round(2)} %.')
                    rivi = rivi+df2.shape[0] + 4
        ws2.write(0, 0, 'Kategoristen muuttujien ristiintaulukoinnit', bold_format)    
    
    # Dikotomisten yhteenveto
    if len(dikotomiset) > 1:
        rivi = 2 # Excelin rivinumero, johon seuraava taulukko kirjoitetaan
        ws3 = workbook.add_worksheet('dikotomiset')
        df1, n = dikot_excel()
        df1.to_excel(writer, sheet_name='dikotomiset', startrow=rivi)
        ws3.write(rivi, 0, '', title_format)
        ws3.write(rivi+df1.shape[0]+1, 2, f'n = {n}', right_format)
        for sarake, arvo in enumerate(df1.columns.values):
            ws3.write(rivi, sarake+1, arvo, header_format)
        for rivinumero, arvo in enumerate(df1.index.values):
            ws3.write(rivi+rivinumero+1 , 0, arvo, index_format)
        ws3.set_column('C:C', 12, cell_format=percent_format)
        ws3.write(0, 0, 'Dikotomisten muuttujien yhteenvedot', bold_format)
    
    # Kategoristen muuttujien ristiintaulukointi dikotomisten kanssa
    if dikotomiset:
        if kategoriset:
            rivi = 3
            ws4 = workbook.add_worksheet('kategoriset+dikotomiset')
            for muuttuja in kategoriset:
                df1, n_arvot = risti_dikot_excel(muuttuja)
                df1.to_excel(writer, sheet_name='kategoriset+dikotomiset', startrow=rivi)
                ws4.write(rivi-1, 1, muuttuja)
                for sarake, n in enumerate(n_arvot):
                    ws4.write(rivi+df1.shape[0]+1, sarake, n)
                ws4.write(rivi, 0, '', title_format)
                for sarake, arvo in enumerate(df1.columns.values):
                    ws4.write(rivi, sarake + 1, arvo, header_format)
                for rivinumero, arvo in enumerate(df1.index.values):
                    ws4.write(rivi+rivinumero+1 , 0, arvo, index_format)
                for i in range(rivi+1, rivi+df1.shape[0]+1):
                    ws4.set_row(i, cell_format=percent_format)
                rivi = rivi + df1.shape[0] + 4
            ws4.write(0, 0, 'Kategoristen muuttujien ristiintaulukoinnit dikotomisten kanssa', 
                      bold_format)
    
    # Tilastolliset tunnusluvut (myös kategoristen muuttujien mukaan ryhmiteltyinä)
    if kvantit:
        rivi = 2
        ws5 = workbook.add_worksheet('tunnusluvut')
        df1 = tunnu()
        df1.to_excel(writer, sheet_name='tunnusluvut', startrow=rivi)
        ws5.write(rivi, 0, '', title_format)
        for sarake, arvo in enumerate(df1.columns.values):
            ws5.write(rivi, sarake+1, arvo, header_format)
        for rivinumero, arvo in enumerate(df1.index.values):
            ws5.write(rivi+rivinumero+1, 0, arvo, index_format)
        if kategoriset:
            rivi = rivi + 11
            for muuttuja1 in kategoriset:
                for muuttuja2 in kvantit:
                    if muuttuja1 != muuttuja2:
                        df1, testi = risti_tunnu_excel(muuttuja1, muuttuja2)
                        df1.to_excel(writer, sheet_name='tunnusluvut', startrow=rivi)
                        ws5.write(rivi-1, 1, muuttuja1)
                        ws5.write(rivi, 0, muuttuja2, title_format)
                        for sarake, arvo in enumerate(df1.columns.values):
                            ws5.write(rivi, sarake+1, arvo, header_format)
                        for rivinumero, arvo in enumerate(df1.index.values):
                            ws5.write(rivi+rivinumero+1 , 0, arvo, index_format)

                        if testi != False:
                            ws5.write(rivi+2, 4, 
                                      f't = {testi[0]:.3f}, p-arvo = {testi[1]:.3f}.')
                        rivi = rivi+11
        ws5.set_column(0, 0, 12)
        ws5.write(0, 0, 
                  'Määrällisten muuttujien tunnusluvut ja tunnusluvut kategoristen ' 
                  'muuttujien määräämissä ryhmissä', bold_format)

    # Korrelaatiot
    if kvantit:
        if len(kvantit) > 1:
            rivi = 2
            ws6 = workbook.add_worksheet('korrelaatiot')
            dfr, dfp, dfn = korre_excel()
            dfr.to_excel(writer, sheet_name='korrelaatiot', startrow=rivi)
            ws6.write(rivi, 0, '', title_format)
            for sarake, arvo in enumerate(dfr.columns.values):
                ws6.write(rivi, sarake+1, arvo, header_format)
            for rivinumero, arvo in enumerate(dfr.index.values):
                ws6.write(rivi+rivinumero+1, 0, arvo, index_format)
            for rivinumero in range(rivi+1, rivi+dfr.shape[0]+1):
                ws6.set_row(rivinumero, cell_format=decimal3_format)
            rivi = rivi + dfr.shape[0] + 2
            dfp.to_excel(writer, sheet_name='korrelaatiot', startrow=rivi)
            ws6.write(rivi, 0, 'p-arvot', title_format)
            for sarake, arvo in enumerate(dfr.columns.values):
                ws6.write(rivi, sarake+1, arvo, header_format)
            for rivinumero, arvo in enumerate(dfr.index.values):
                ws6.write(rivi+rivinumero+1, 0, arvo, index_format)
            for rivinumero in range(rivi+1, rivi+dfp.shape[0]+1):
                ws6.set_row(rivinumero, cell_format=decimal3_format)
            rivi = rivi + dfr.shape[0] + 2
            dfn.to_excel(writer, sheet_name='korrelaatiot', startrow=rivi)
            ws6.write(rivi, 0, 'n-arvot', title_format)
            for sarake, arvo in enumerate(dfr.columns.values):
                ws6.write(rivi, sarake+1, arvo, header_format)
            for rivinumero, arvo in enumerate(dfr.index.values):
                ws6.write(rivi+rivinumero+1, 0, arvo, index_format)
            ws6.write(0, 0, 'Määrällisten muuttujien väliset korrelaatiokertoimet', 
                      bold_format)
    
    writer.close()
    print(f'Tulokset löytyvät Excel-tiedostosta {tiedosto}.')