# Automatisoi data-analytiikkaa

Moduuli **ana.py** sisältää helppokäyttöisiä funktioita kuvailevaan ja selittävään analytiikkaan. Funktioiden käytön opit **ana.ipynb**-muistion esimerkeistä. Lataa tämä repository omalle koneellesi, niin pääset käyttämään funktioita, jos koneellesi on asennettu Python, Jupyter Notebook, numpy, pandas ja xlwings. Jos et ole aiemmin ladannut repostitorya githubista, niin lue ohje https://tilastoapu.wordpress.com/2018/11/27/github-esimerkit-omalle-koneelle/.

## Laske keskeiset kuvailevan ja selittävän analytiikan tulokset suoraan Excel-tiedostoon

Muistiosta **ana.ipynb** löydät esimerkin **ana_excel**-funktion käytöstä. Funktio **ana_excel** laskee ja kirjoittaa Excel-tiedostoon omille taulukkovälilehdilleen siististi muotoiltuna
- kategoristen muuttujien frekvenssitaulukot
- kaikki mahdolliset kahden kategorisen muuttujan ristiintaulukoinnit ja khiin neliö -testit
- dikotomisten muuttujien yhteenvedon
- kaikkien kategoristen muuttujien ristiintaulukoinnit dikotomisten muuttujien kanssa
- kvantitatiivisten muuttujien tilastolliset tunnusluvut, kvantitatiivisten muuttujien tilastolliset tunnusluvut kategoristen muuttujien mukaan ryhmiteltynä ja kahden ryhmän vertailutestit (t-testit)
- korrelaatiokertoimet ja niiden merkitsevyyden testauksen.


