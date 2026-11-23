# **Roomor · Contesto del progetto** 

Nov 17, 2026 · @Filippo · aggiornato il 23 nov 2026 (regola fissa per recensioni scritte come testo unico, test su un hotel svizzero mai visto, controllo a mano, branch `flavio-pre-web-development`)

Questo documento riassume tutto il progetto Roomor: cosa abbiamo fatto, cosa stiamo facendo e come lavoriamo. Per riprendere il lavoro in una chat nuova, incolla il link a questo documento (oppure esportalo in Markdown e allegalo) e scrivi "leggi questo contesto prima di rispondere". Va aggiornato a ogni passo importante. 

## **Il corso e le regole** 

Roomor è il progetto semestrale del corso "Introduction to Machine Learning and Deep Learning" (iMLDL) all'Università di San Gallo (HSG), tenuto dal Prof. Bruno Rodrigues. Il compito è fondare una startup fittizia con un prodotto basato su machine learning. 

- **Team:** Filippo Strafile e Flavio Massimo Campanile. Entrambi devono capire tutto il progetto e parlare in tutte le presentazioni. 

- **Voto del progetto (60% del corso):** mid-term 15% (fatto il 20 ottobre), pitch finale 20%, codice 15%, sito web 10%; extra come bonus (video, demo interattiva, feedback utenti). L'esame scritto vale l'altro 40%. 

- **Scadenze:** **tutto va consegnato entro l'8 dicembre** (settimana 11, feature freeze): codice, demo, sito, slide finali ed extra. Il Gruppo 1 presenta il **15 dicembre** (pitch di circa 10 minuti con demo dal vivo). 

- **Obbligatorio:** una sezione "Trustworthy AI" con limiti, rischi e usi impropri, bias dei dati. 

- **Uso dell'AI:** permesso per scrivere codice, ma ognuno di noi deve saper spiegare qualsiasi riga dal vivo. 

- **Dati:** pubblici, con licenza o nostri; vanno citati. Niente scraping di Booking: i suoi termini (sezione A15.2) vietano la raccolta automatica. 

## **L'idea: Roomor** 

Roomor legge le recensioni di un hotel e dice al direttore cosa sistemare per primo, ordinando i problemi per quanto costano in voto degli ospiti. Slogan: _"Your guests talk. We translate."_ 

- **Problema:** un hotel riceve centinaia di recensioni su più piattaforme e in più lingue. Il voto medio (8.4/10 nel 515K) dice quanto sono soddisfatti gli ospiti, non quale parte dell'hotel li delude. Conseguenza: il budget per ristrutturazioni e investimenti rischia di andare nel posto sbagliato. 

- **Utente:** direttori generali e proprietari di hotel indipendenti e piccoli gruppi, senza un team interno di analisi. Partiamo dalla Svizzera. Nello stesso hotel la stessa persona legge le recensioni, decide il budget e firma il contratto. 

- **Cosa entra:** il testo delle recensioni dell'hotel. Nessuna installazione, i modelli non si riaddestrano per ogni hotel. 

- **Cosa esce:** un report con i temi ordinati per impatto. Per ogni tema: quanto spesso gli ospiti se ne lamentano, quanti punti di voto costa in media una critica (e quanti ne aggiunge un 

complimento), e 2–3 frasi vere come esempio. Priorità di un tema = (quota di recensioni che lo criticano − quota nell'hotel medio) × peso del tema; si mostrano anche i punti persi in assoluto (quota × peso). 

- **Concorrenti:** TrustYou, Customer Alliance, MARA. Differenza ipotizzata, da verificare con gli albergatori: priorità per impatto sul voto e non per frequenza, e trasparenza (ogni tema rimanda alle frasi da cui viene). 

- **Prova del problema:** studio Cornell (Anderson, 2012): un punto in più su una scala da 5 permette di alzare i prezzi di circa l'11% a parità di occupazione. Studio datato, da dichiarare. 

- **Perché questa idea:** la prima idea (domanda di pane per le panetterie) è stata scartata dal prof perché richiedeva di riaddestrare il modello per ogni cliente. Le recensioni invece sono un input universale: un modello addestrato una volta funziona per qualsiasi hotel. 

## **La pipeline** 

Tre modelli supervisionati, addestrati una volta sola su dati pubblici, più una regola fissa: ogni recensione dell'hotel cliente li attraversa e diventa una lista ordinata. 



<!-- Start of picture text -->
Each review flows througha fixed rule and three models into a ranked list<br>Top: what happens to every review. Bottom: the public data that trains the models once.<br>Model 1<br>topics<br>Review Fixed rule Model 3 Ranked list<br>plain text splits pieces score impact what to fix first<br>Model 2<br>sentiment<br>trains 1 and 2 trains 3<br>Trained once on public data BEAST SIsiBeeklig,<br>23,113 sentences 515,738 reviews<br><!-- End of picture text -->

pipeline · 1 rule, 3 models, 2 datasets 

|Pezzo|Cosa fa|Algoritmo|Baseline da<br>battere|Metrica|Stato|
|---|---|---|---|---|---|
|Regola fissa|Spezza una recensione scritta come testo unico su . ! ? ; e a capo, prima di but, however, although, though, except, whereas, unfortunately, e sulle virgole se un pezzo ha due temi; il Model 2 manda ogni pezzo al lato negativo o positivo (soglia 0.50, parola di contrasto tolta prima del Model 2). Nel 515K (campi Booking già separati) non serve: lì restano le finestre di 12 parole|Regola scritta a<br>mano + Model 2|Tutto nel lato negativo; Model 2 senza tagli|Accordo con i campi Booking (F1 temi, stesso n. 1 in Fix first); pezzi dalla parte giusta|**Fatto (23 nov)**: 515K stesso n. 1 nell'82% degli hotel (baseline 41% e 5%); citizenM Zürich Fix first identico ai campi Booking|



|Pezzo|Cosa fa|Algoritmo|Baseline da<br>battere|Metrica|Stato|
|---|---|---|---|---|---|
||parole|||||
|Model 1 ·<br>temi|Quali dei 21 temi cita<br>un pezzo di frase|Classificazione<br>multi-etichetta:<br>21 regressioni<br>logistiche su TF-<br>IDF, soglia per<br>tema|Parole<br>chiave<br>scritte a<br>mano;<br>"sempre<br>Room"|F1 per tema,<br>micro e<br>macro F1|**Fatto (17**<br>**nov), micro**<br>**F1 0.857**|
|Model 2 ·<br>sentiment|Positivo o negativo,<br>con probabilità|Classificazione<br>binaria:<br>regressione<br>logistica su TF-<br>IDF|Sempre<br>"positive";<br>parole<br>chiave|Accuracy, F1|Fatto (da<br>rifare con<br>validation)|
|Model 3 ·<br>impatto|Punti di voto persi o<br>guadagnati per tema|Regressione<br>lineare multipla,<br>temi criticati e<br>lodati come<br>colonne 0/1|Voto medio<br>per ogni<br>recensione|MAE, R²,<br>bootstrap|Fatto su<br>tutto il 515K<br>(17 nov):<br>MAE 0.97 su<br>hotel mai<br>visti; 37 pesi<br>su 42<br>affidabili al<br>bootstrap<br>per hotel|



Per allenare il Model 3, prima si applica il Model 1 ai campi Negative_Review e Positive_Review del 515K: ogni recensione diventa una riga di temi 0/1 più il voto. Il sentiment del Model 2 serve solo in fase di uso, sulle recensioni dei clienti, che non sono già divise in positivo e negativo. 

## **I dati** 

Usiamo due dataset pubblici di Kaggle: HRAST insegna ai Model 1 e 2 a leggere le frasi, il 515K collega il testo al voto per il Model 3. 

||HRAST|515K Booking.com reviews|
|---|---|---|
|Fonte|Kaggle<br>,CC BY-NC 4.0 (non<br>commerciale)|Kaggle<br>,CC0; dati originari di Booking.com|
|Dimensione|23,113 frasi, 28 colonne|515,738 recensioni, 17 colonne, 238 MB|
|Contenuto|Frasi da recensioni Booking di 42 hotel<br>(Napoli,Salisburgo,Barcellona,|1,492 hotel in 6 città (London, Paris,<br>Barcelona,Amsterdam,Vienna,Milan),|



||HRAST|515K Booking.com reviews|
|---|---|---|
||Copenaghen), solo inglese|agosto 2015 – agosto 2017|
|Etichette|21 temi 0/1; sentiment positive 11,819<br>(51%) / negative 10,504 (45%) /<br>neutral 794 (3%); Aspect = 1 in 2,884<br>frasi con più temi (non "miste": vedi 23 nov)|Voto 2.5–10 (media 8.4, mediana 8.8, 48% ≥<br>9); testo diviso in Negative_Review e<br>Positive_Review; nessuna etichetta di tema|
|Serve a|Model 1 (temi) e Model 2 (sentiment)|Model 3 (impatto sul voto)|
|Problemi|BOM nel file (<br>encoding="utf-8-sig"); a<br>capo dentro le virgolette (Excel spezza<br>le righe); 8 duplicati; 4 righe con due<br>sentiment; "σ" nella colonna Location<br>(id 4634); nessuna colonna hotel o<br>città|526 duplicati; testo senza punteggiatura;<br>"nessuna critica" scritto in molti modi ("No<br>Negative" 127,890 volte, "Nothing", "N A"…,<br>circa 30%); lat/lng "NA" in 3,268 righe; date<br>mese/giorno/anno; 48% dei recensori dal<br>Regno Unito|



Temi più frequenti in HRAST: Room 5,004, Location 4,729, Staff 3,804. Nel 515K l'hotel si identifica con <mark>Hotel_Address</mark> (1,493 indirizzi per 1,492 nomi). 

**Campione da 300 recensioni (file** **<mark>WWRSF_HotelReviewRegression.xlsx</mark> ).** 300 recensioni vere estratte a caso dal 515K (seed 42) tra le 474,087 che citano almeno uno di 5 temi: Staff, Room, Noise, Location, Restaurant. Le colonne <mark>Neg_<tema></mark> e <mark>Pos_<tema></mark> valgono 1 se nel testo negativo o positivo compare una parola chiave del tema. **I temi sono trovati con parole chiave, non a mano** : va sempre detto così. Le parole chiave sbagliano ("no towels left after service" contato come critica allo Staff). 

**Catalogo degli hotel svizzeri (23 nov, file `data/Hotel_Database.xlsx`, su GitHub).** 40 hotel catalogati da Flavio, candidati per il test set svizzero 10 × 30. Colonne: `ID_Hotel`, `Name_Hotel`, `City`, `Cluster`, `Stars (Booking)`, `Restaurant`, `Wellness`, `Parking`, `Ski` (0/1), `Stato` (vuota). Per cluster: Urban/Business 26, Alpine Resort 9, Lakes/Mountain 5. Per stelle: 4★ 21, 3★ 17, 5★ 2. Servizi: ristorante 22, wellness 21, parcheggio 23, sci 8. Il primo hotel (H001, citizenM Zürich) è quello già usato nel test del 23 novembre.

- **Zurigo (7):** H001 citizenM Zürich (4★), H005 Olympia Hotel Zurich (3★), H018 Hotel Limmatblick (3★), H019 Hotel Hirschen (3★), H020 Storchen Zürich (5★), H021 Placid Hotel Design (4★), H040 Crowne Plaza Zürich by IHG (4★)
- **San Gallo (9):** H004 B&B HOTEL St Gallen (3★), H014 Hotel Walhalla (4★), H015 Hotel Dom (3★), H016 Radisson Blu Hotel (4★), H017 Oberwaid (4★), H034 Militärkantine St. Gallen (3★), H035 Sorell Boutique-Hotel City Weissenstein St Gallen (4★), H036 Einstein St. Gallen (4★), H037 Tailormade Hotel METROPOL St Gallen (4★)
- **Ginevra (5):** H007 Nash Airport Hotel (4★), H008 Hilton Geneva Hotel (4★), H013 Warwick Geneva (4★), H025 ibis Genève Centre Nations (3★), H026 ibis Geneve Centre Lac (3★)
- **Berna (3):** H022 Nydeck (3★), H023 Swissôtel Kursaal Bern (4★), H024 NH Bern The Bristol (4★)
- **Losanna (2):** H030 Hotel Mirabeau (4★), H031 Swiss Wine by Fassbind (3★)
- **Zermatt (8, Alpine Resort):** H002 Riffelalp Resort (5★), H006 Matterhorn Inn (3★), H009 Matterhorn Lodge Boutique Hotel (4★), H010 Schweizerhof Zermatt (4★), H011 Hotel Capricorn (3★), H012 Hotel Daniela (4★), H038 Hotel Bristol (3★), H039 Hotel Continental (3★)
- **Crans-Montana (1, Alpine Resort):** H003 Hôtel Splendide (3★)
- **Lugano (3, Lakes/Mountain):** H027 Suitenhotel Parco Paradiso (4★), H028 Hotel Colibrì (3★), H029 Villa Sassa (4★)
- **Lucerna (2, Lakes/Mountain):** H032 Hotel Seeburg (4★), H033 Hotel Royal Luzern (3★)

**Test set svizzero:** da costruire. Pilota di circa 10 hotel × 30 recensioni in inglese, etichettate a mano da noi due; serve l'approvazione del prof per la copia manuale. 

## **Cosa abbiamo fatto finora** 

Al mid-term del 20 ottobre avevamo il sentiment (Model 2) e una prima versione del Model 3 su 300 recensioni. Il 17 novembre abbiamo costruito il Model 1 (temi): ora i tre modelli esistono tutti, manca il Model 3 su tutto il 515K. 

### **Aggiornamento 17 novembre, pomeriggio.** 

- **Model 1 applicato al 515K** <mark>(src/apply_topics.py</mark> → <mark>data/review_topics.csv</mark> , 515,212 recensioni dopo i duplicati su tutte le colonne). Testi divisi in finestre di 12 parole con passo 6. Segnaposti di "nessuna critica" svuotati con due regole: testi identici a una lista ("No Negative", "nothing", "n a", "leaving"…) ed espressioni brevi ("can t fault", "no complaints"…) solo in testi di al massimo 8 parole senza but / except / apart / only / however / although / besides. 

- **Filtro con il sentiment provato e scartato.** L'idea era scartare le finestre del campo negativo che il Model 2 giudica lodi. Su un controllo a mano scartava più critiche vere che lodi (Location: 13 sbagliate su 20), perché il Model 2 ha imparato parole di tema come "location" e "clean" come positive, le finestre tagliano le negazioni e il modello non conosce il neutro. Lezione: il 94.6% di 

HRAST non vale su dati diversi. Una regola mirata su "nothing…" è tenuta disattivata, per un'analisi di sensibilità. 

• **Model 3 su tutto il 515K** <mark>(src/impact_full.py</mark> ): divisione per hotel (1,269 hotel / 432,531 recensioni in train, 224 hotel / 82,681 recensioni in test, nessun hotel in comune). Su hotel mai visti: MAE 0.97 contro 1.29 della baseline (voto medio del train, 8.37), R² 0.36 contro −0.009. Intercetta 8.34. Pesi principali: critiche a Staff −0.80, Clean −0.70, Room −0.61, Bed −0.50, Bathroom −0.46, Noise −0.42; lodi a Staff +0.68, Generic +0.47, View +0.28. Asimmetria tipo Kano: Clean −0.70 / +0.08, Bathroom −0.46 / +0.10, Value for money −0.39 / +0.06; View −0.01 / +0.28. Location lodata nel 48.5% delle recensioni ma vale solo +0.17. Lodi a Lift, Reception e Wi-Fi con segno negativo: temi rari, da verificare con il bootstrap. 

### **Aggiornamento 22 novembre.** 

- **Bootstrap per hotel** (200 ricampionamenti dei 1,269 hotel di train, circa 8 minuti): 37 pesi su 42 affidabili. Non affidabili: Neg_Beach, Neg_View, Pos_Parking, Pos_Beach, e Neg_Bar al limite (con 1,000 ripetizioni l'intervallo tocca lo zero). Pesi principali molto precisi: Neg_Staff da −0.83 a −0.77, Pos_Staff da +0.66 a +0.70. Ricampionare per hotel invece che per recensione rende gli intervalli in media 1.66 volte più larghi (3.4 volte per Pos_Staff): le recensioni dello stesso hotel non sono indipendenti. Pos_Lift (−0.21), Pos_Reception (−0.07) e Pos_Wi-Fi (−0.03) sono affidabili ma con segno controintuitivo: nel report non vanno mostrati tra le cose da proteggere. 

- **Pesi salvati** in <mark>models/impact_weights.csv</mark> (colonne topic, coef, lower, upper, share_pct; share_pct calcolata sul train); intercetta (8.338) in <mark>models/impact_intercept.json</mark> . 

- **Confronto con le 300 recensioni:** segno uguale in 10 pesi su 10, 9 su 10 dentro l'intervallo della prova; nel modello completo i pesi sono quasi tutti più vicini a zero (Neg_Room da −1.10 a −0.61), perché con 42 colonne l'effetto della camera si divide tra Room, Bed, Bathroom, Clean e Comfort, e perché gli errori del Model 1 attenuano i pesi. 

- **Analisi di sensibilità sulla regola "nothing"** <mark>(experiments/impact_nothing.py</mark> ): la regola cambia 582 recensioni su 515,212 (0.11%, voto medio 9.63); differenza massima tra i pesi 0.006 (Neg_Generic, da −0.453 a −0.460); MAE 0.9685 contro 0.9681. Decisione: si tiene la tabella senza regola. 

- **Decisione:** DistilBERT dopo il report, con un limite di 2 giorni, sul Model 1 (temi) perché è quello che costruisce la tabella del Model 3. Successo = batte micro F1 0.857 / macro 0.852 su HRAST e poi il MAE 0.97 su hotel mai visti. 

### **Aggiornamento 22 novembre, struttura della repo (branch `main`).**

- **Si lavora su `main`.** La riorganizzazione fatta nella branch `submission` (commit `cbebde2`) è stata unita in `main` insieme a `pipeline.py` e `report.py`. I dataset, caricati per errore su `submission` (commit `cb17823`), **non** sono in `main`. La branch `submission` è stata cancellata da GitHub il 22 novembre: non farne più il push, rimetterebbe i dataset online. Il commit `cb17823` può restare raggiungibile con il link diretto per un po' e `Hotel_Reviews.csv` resta in Git LFS: per toglierli del tutto serve il supporto GitHub (proprietario della repo).

- **Nuova struttura:** 
  - `src/` · la pipeline finale: `topic_model.py`, `sentiment_model.py`, `apply_topics.py`, `impact_full.py`, `pipeline.py` (funzioni condivise, si importa e non si esegue) e `report.py` (report per un hotel), `verify_hotels.py` (verifica sui 224 hotel). 
  - `experiments/` · versioni precedenti e analisi di sensibilità, fuori dalla pipeline finale: `impact_model.py` (Model 3 sulle 300 recensioni, mid-term) e `impact_nothing.py` (regola "nothing"). 
  - `app/` · la demo dal vivo (per ora solo un README). 
  - `docs/` · `CODE_EXPLANATION.md`, `TRUSTWORTHY_AI.md` (bozza con i titoli: limiti, rischi, usi impropri, bias dei dati) e `docs/internal/` con questo file e `To-Do-List.md`. 
  - `data/` (sempre minuscolo: su Mac e Linux `Data` è un'altra cartella) e `models/` · ognuna ha un `README.md` tracciato che dice quali file metterci e da quale script nascono. **Aggiornato il 23 nov:** vedi "Cosa va su GitHub" qui sotto. 
  - Nella radice: `README.md` riscritto in inglese per la consegna (nome Roomor, struttura, pipeline, come riprodurre i risultati, dati e licenze), `requirements.txt` con le versioni usate, `.gitignore` aggiornato. 

- **Come si lanciano ora gli script:** per esempio `python src/topic_model.py` dalla cartella principale. Ogni script trova `data/` e `models/` partendo dalla propria posizione, quindi funziona da qualunque cartella. Controllato: tabella su 1,000 recensioni identica, report identico, Model 1 e 2 riaddestrati identici.

- **Da completare nella struttura:** il testo di `docs/TRUSTWORTHY_AI.md`, la demo in `app/`. I link Kaggle e la citazione di HRAST sono nel README. Il sito web non ha ancora una cartella: da decidere dove ospitarlo.

### **Aggiornamento 22 novembre, generatore di report (`src/report.py`).**

- **Cosa fa:** `make_report(negative_texts, positive_texts)` riceve le recensioni di un hotel e restituisce "Fix first" e "Protect". Usa solo i 37 pesi affidabili (Neg_Bar escluso a mano). Prima di cercare i temi, `normalize_text` porta i testi nel formato del 515K (senza punteggiatura).

- **Ordinamento rispetto all'hotel medio:** Fix first è ordinato per (quota dell'hotel − quota media) × peso. La quota media è share_pct (train). Un tema entra solo se supera la media di almeno 5 punti percentuali e lo criticano almeno 5 recensioni; altrimenti "No topic is clearly worse than the average hotel". Motivo: con i punti assoluti Staff e Room salivano ai primi posti in tutti e 3 gli hotel provati, anche in quello con il voto più alto.

- **Generic escluso da Fix first** (resta nella regressione e può comparire in Protect): nel controllo a mano (etichette di Claude, vedi sotto) 13 critiche su 19 contate come Generic erano sbagliate.

- **Protect:** i 3 temi lodati che fanno guadagnare più punti, solo con peso positivo e affidabile.

- **Esempi:** 2 finestre di almeno 4 parole per tema, scelte con il Model 2 (più negative per Fix first, più positive per Protect), mai la stessa finestra in due temi. Il sentiment sceglie solo gli esempi, non cambia i conteggi: gli esempi possono quindi sembrare più precisi dei numeri. Avviso sotto le 30 recensioni.

- **Prova su 3 hotel mai visti:** Savoy Amsterdam (voto 6.01): 9 temi in Fix first, primo Room (−0.137). Mövenpick Amsterdam (8.51): Value for money, Location, Check-in. H10 Casa Mimosa (9.66): nessun tema chiaramente peggiore della media.

- **Errore di conteggio:** 25 recensioni a caso per hotel (seed 42) tra quelle con almeno un tema criticato, 114 temi contati: **circa 30% sbagliati** (34 su 114), **12% al Savoy** (7 su 57) e **47% al Casa Mimosa** (27 su 57); altri 15 dubbi. Non è la media di tutti gli hotel: sono i due estremi. **Etichette messe da Claude, da ricontrollare a mano da noi due prima di dichiararle** (file `data/count_check_labels.csv`, fuori da GitHub).

### **Aggiornamento 22 novembre, verifica sui 224 hotel di test (`src/verify_hotels.py`).**

- **Metodo:** voto previsto di un hotel = intercetta + somma sui 42 temi di (quota × peso), confrontato con il voto medio vero. Hotel mai visti dal Model 3 (82,681 recensioni). Grafico in <mark>results/hotel_scores.png</mark> .

- **Risultati:** correlazione 0.907; MAE per hotel 0.310 contro 0.523 della baseline (sempre 8.37), cioè 41% in meno; per singola recensione il miglioramento è 25% (0.968 contro 1.288). Il passaggio da 0.97 a 0.31 è in parte solo l'effetto della media: il confronto giusto è con la baseline.

- **Compressione delle differenze:** terzo di hotel con voto più basso previsto troppo alto (+0.37), terzo con voto più alto previsto troppo basso (−0.41); deviazione standard dei voti previsti 0.299 contro 0.641 dei veri. Caso peggiore Savoy: 7.50 previsto, 6.01 vero. L'ordine tra hotel è affidabile, l'ampiezza no: i "punti persi" del report sono priorità relative, non punti di voto recuperabili.

- **Spiegazione probabile (non verificata):** attenuation bias dovuto agli errori del Model 1: colonne 0/1 con errori spingono i pesi verso zero, quindi le previsioni si stringono verso la media. Possono contribuire anche la gravità delle critiche (il modello sa se un tema è criticato, non quanto) e ciò che non è un tema.

- **Cosa non dimostra:** che i singoli temi siano giusti (gli errori possono compensarsi), che sistemare un problema alzi il voto, che valga per hotel svizzeri o in altre lingue. Temi e voto vengono dalle stesse recensioni: non è una previsione del futuro.

### **Aggiornamento 22 novembre, DistilBERT sul Model 1 (temi).**

- **Metodo:** `src/export_split.py` salva la stessa divisione del Model 1 (16,173 / 3,466 / 3,466 frasi, seed 42). Fine-tuning di distilbert-base-uncased su Google Colab (Tesla T4, torch 2.14.1+cu130, transformers 5.22.1 nel secondo addestramento): 21 uscite con sigmoide, learning rate iniziale 2e-5 che scende a 0, batch 32, 64 token, seed 42. Quaderni in `notebooks/`. Una soglia per tema scelta sul validation con la stessa regola del Model 1. Confronto in `experiments/compare_distilbert.py`.

- **Il test è stato usato QUATTRO volte: 3 epoche (seed 42), 8 epoche (seed 42), 8 epoche (seed 1) e 8 epoche (seed 2).** Le ultime due solo per applicare una regola già scritta, senza scegliere nulla. Prima dei nuovi seed era stato guardato due volte: la decisione di allungare l'addestramento è stata presa dopo aver visto il primo risultato sul test, quindi il secondo può essere un po' ottimista. Lo stesso limite sui temi rari si vedeva anche sul validation. Da dichiarare nel pitch e in Trustworthy AI.

- **Prima volta, 3 epoche:** test micro F1 0.8733 / macro F1 0.8445. Micro meglio del TF-IDF, macro peggio (differenza −0.007, intervallo da −0.019 a +0.004): criterio (batterli entrambi) non raggiunto. Crolli sui temi rari: Restaurant 0.674 → 0.462, Wi-Fi 0.946 → 0.816.

- **Seconda volta, 8 epoche, si tiene l'epoca con il macro F1 più alto sul validation (epoca 7):** validation micro 0.8952 / macro 0.8872 (TF-IDF 0.8615 / 0.8680); test **micro F1 0.8898 / macro F1 0.8774** contro 0.8567 / 0.8517 del TF-IDF. Differenze +0.033 (intervallo 95% da +0.027 a +0.039) e +0.026 (da +0.014 a +0.036), bootstrap di 1,000 ricampionamenti delle frasi di test. Temi aggiunti per errore 879 → 608, temi mancati 741 → 621, frasi tutte giuste 63.8% → 72.9%. Meglio in 20 temi su 21 (Generic 0.625 → 0.736); peggio Restaurant (0.674 → 0.644, solo 53 frasi). Dall'epoca 4 in poi il macro F1 sul validation cambia poco (0.884–0.887).

- **Limiti:** l'intervallo considera solo il caso nelle frasi di test; i tre seed usano sempre la stessa divisione dei dati. HRAST non è il 515K: non sappiamo ancora se il report migliora. Il Model 1 riconosce il tema, non se il commento è negativo, quindi le critiche contate per errore possono restare. Il modello pesa 268 MB ed è lentissimo senza GPU.

- **Regola decisa il 22 novembre, prima di vedere i risultati:** si rifà l'addestramento a 8 epoche con seed 1 e seed 2, senza altre modifiche. DistilBERT si tiene solo se **tutti e tre i seed** (42, 1, 2) battono 0.857 / 0.852 sul test. Con i seed il test viene calcolato altre due volte: da dichiarare.

- **Risultato della regola (22 novembre): DistilBERT si tiene, tutti e tre i seed battono 0.857 / 0.852.** Test micro / macro F1: seed 42 0.8898 / 0.8774 (epoca 7), seed 1 0.8908 / 0.8780 (epoca 7), seed 2 0.8901 / 0.8832 (epoca 8). **Media 0.8903 / 0.8795, minimo 0.8898 / 0.8774.** Per tutti i seed gli intervalli al 95% della differenza con il TF-IDF non contengono lo zero. 17 temi su 21 migliorano con tutti e tre i seed; in media peggiorano solo Lift (−0.011, 43 frasi) e Restaurant (−0.007, 53 frasi), differenze incerte. Errori totali da 1,620 a 1,229 / 1,227 / 1,225. Stesse versioni per tutti gli addestramenti a 8 epoche (torch 2.14.1+cu130, transformers 5.22.1, Tesla T4). Quaderno: `notebooks/distilbert_topics_seeds.ipynb`.

- **Modello da usare: seed 42** (cartella Drive Roomor/distilbert_topics_8epochs/model). Il seed 2 ha il macro F1 più alto, ma sceglierlo dopo aver visto il test sarebbe proprio l'errore da evitare, e la differenza è minima. Nel pitch si riportano media e minimo dei tre seed.

- **Prossimi passi:** noi due ricontrolliamo le 114 etichette; DistilBERT sulle 50 recensioni per vedere se conta meno critiche sbagliate; tempo su un campione del 515K, poi si decide se applicarlo a tutto. Secondo criterio: battere il MAE 0.97 su hotel mai visti (non ancora provato). 

### **Aggiornamento 23 novembre (Flavio, branch `flavio-pre-web-development`).**

Domanda di partenza: il prodotto funziona con recensioni nuove di un hotel mai visto, anche se non sono divise in "liked" / "disliked" come su Booking (Google, TripAdvisor)? Risposta: sì, a certe condizioni. Tutti i numeri sotto sono stati rigenerati sul PC di Flavio con risultati identici a quelli di Claude.

- **Modelli rigenerati** sul PC di Flavio (`models/` era vuota): Model 1 0.8567 / 0.8517, Model 2 accuracy 0.9458, Model 3 MAE 0.9685, 4 pesi non affidabili su 42. Identici al 22 novembre (scikit-learn 1.8.0, Python 3.12 di VS Code: serve `python -m pip install -r requirements.txt`).
- **Regola fissa costruita** (`src/pipeline.py`: `split_review`, `route_pieces`, `split_full_text`; `src/report.py`: `make_report_from_full_texts`). Le funzioni esistenti non sono cambiate: report dei 3 hotel di prova identico riga per riga.
- **Soglie scelte su 150 hotel di train** (`experiments/choose_thresholds.py`), con una regola decisa prima: si cambia solo se la sovrapposizione dei temi di Fix first con i campi Booking migliora di più di 0.01. Risultato: **0.50 / 0.50** (nessun pezzo messo da parte: con 0.35 / 0.65 si perdevano più critiche di quante se ne salvassero; 0.817 → 0.850) e parola di contrasto tolta prima del Model 2 (effetto minimo: 0.850 contro 0.847). L'idea di ottobre di "mettere da parte i pezzi incerti" è abbandonata.
- **Test sui 224 hotel, guardato una volta dopo la scelta** (`src/verify_full_texts.py`): "liked" e "disliked" del 515K incollati in un testo unico. Con un punto tra le due parti: F1 temi criticati 0.921, lodati 0.951; stesso n. 1 in Fix first nell'**81.7%** degli hotel (Model 2 senza tagli 40.6%, tutto nel lato negativo 5.4%); correlazione voto previsto / vero 0.896 (campi Booking 0.907). Senza punteggiatura la regola aiuta poco (stesso n. 1 nel 50%). HRAST con punteggiatura: 92.6% delle frasi dalla parte giusta (senza tagli 67.9%). Unico numero che non migliora: MAE per hotel 0.314 contro 0.301 del Model 2 senza tagli, che però sbaglia il n. 1 nel 59% degli hotel.
- **Correzione:** in HRAST **Aspect = 1 vuol dire "più temi nella frase", non "lode e critica insieme"** ("Great location and clean rooms"). Sulle 864 frasi con Aspect = 1 e una parola di contrasto la regola taglia il 97.5% e separa lode e critica nel 59.8%.
- **Hotel svizzero mai visto: citizenM Zürich**, 30 recensioni Booking (lug–nov 2026) copiate a mano da Flavio, in `data/citizenm_zurich_booking.csv` (senza nomi degli ospiti, fuori da GitHub; l'OK del prof per la copia manuale non serve, decisione del 23 nov; 21 con i due blocchi distinguibili). Script `src/check_booking_hotel.py`: 91.7% dei pezzi dalla parte giusta; F1 temi 0.925 / 0.930; **Fix first (Room, Noise, Facilities, Comfort) e Protect (Staff, Hotel overall, Location) identici** a quelli costruiti dai blocchi Booking.
- **Model 3 su citizenM** (`src/explain_hotel.py`, grafico `results/citizenm_zurich_points.png`): voto medio previsto **8.17 contro 8.33 vero** (errore 0.16; sui 224 hotel l'errore medio era 0.31). Per singola recensione invece MAE 0.64 contro 0.58 della baseline "sempre 8.37", correlazione 0.48: va usato solo il dato aggregato. Punti persi −0.85 (Room −0.22, Cleanliness −0.09, Bed −0.08, Noise −0.08), guadagnati +0.69 (Staff +0.25).
- **Controllo a mano** (`src/check_pieces.py` → `data/manual_check.xlsx`; **etichette messe da Claude**, in `data/manual_check_CLAUDE_labels.xlsx`, riassunto in `results/manual_check_summary.md`, **da rivedere da noi** e da dichiarare come "pre-etichettate con un modello AI e riviste da noi"): citizenM 118 pezzi, tagli giusti 93.2%, lato giusto 92.8%, **critiche nel lato negativo 46 / 46**; tutti gli errori sono lodi o fatti neutri letti come critiche (circa 21% dei pezzi negativi; esempio "didn't hear a thing from inside our room" contato sotto Noise). Dei 30 pezzi "sbagliati" del 515K solo il 47% sono errori veri (37% errori dell'ospite nel campo Booking). Frasi HRAST con "but": 23 / 30 giuste.
- **Quante recensioni servono** (prova veloce di Claude con uno script temporaneo, non nella repo): 16 hotel di test con almeno 800 recensioni, 60 campioni per dimensione. Stesso n. 1 del report completo: 30 recensioni 53%, 100 69%, 200 80%, 300 85%, 500 91%. Il voto medio si stabilizza già a 100 (errore 0.08). Proposta: ultimi 12–18 mesi, almeno 100 recensioni.
- **Report di punti di forza e debolezza**, con le informazioni da aggiungere: documento "Roomor · Punti di forza e debolezza" (link nella sezione File e link).
- **Prossimi passi:** Filippo rivede codice e numeri, poi fa il merge in `main`; rivedere le etichette di Claude; poi si segue l'ordine della To-Do list (`docs/internal/To-Do-List.md`).

|Quando|Cosa|Risultato|
|---|---|---|
|17 nov|Model 1 · temi su HRAST<br>(<br>src/topic_model.py): 23,105 frasi dopo i<br>duplicati, divise 16,173 / 3,466 / 3,466<br>(train / validation / test, seed 42); 21<br>regressioni logistiche su TF-IDF; soglia per<br>tema scelta sul validation (quasi tutte tra<br>0.05 e 0.30)|Micro F1 0.857 / macro F1 0.852 sul test.<br>Baseline: parole chiave 0.808 / 0.807,<br>"sempre Room" 0.167 / 0.017. Con soglia<br>0.5 il modello era peggio delle parole chiave<br>(0.797 / 0.727). Vince su 15 temi su 21,<br>soprattutto Generic (0.16 → 0.62), Facilities<br>(0.59 → 0.79), Comfort (0.71 → 0.82); le<br>parole chiave vincono su Clean, View,<br>Beach, Bed, Lift, Value for money|



|Quando|Cosa|Risultato|
|---|---|---|
|16 nov|Messaggio a un contatto nel mondo hotel<br>management per un feedback (utilità da<br>1 a 10, come decidono oggi)|In attesa di risposta|
|10 nov|Feedback scritto del prof sul mid-term|Ricevuto per mail il 16 nov e letto il 23 nov (vedi Domande aperte)|
|20 ott|Mid-term talk, 10 slide in PDF, 8 minuti +<br>2 di domande|Fatto; voto non ancora noto|
|19 ott|Model 3 di prova: regressione lineare<br>sulle 300 recensioni, 10 colonne (5 temi<br>criticati / lodati)|MAE 1.01 contro 1.28 della baseline (voto<br>medio), R² 0.33, cross-validation a 5 fold|
|19 ott|Bootstrap del Model 3 (1,000<br>ricampionamenti delle 300 righe)|Affidabili: critiche a Room −1.10, Staff −1.01,<br>Location −0.88; lodi a Staff +0.89,<br>Restaurant +0.41. Non affidabili: Noise e le<br>altre lodi|
|19 ott|Model 2: TF-IDF + regressione logistica su<br>HRAST (train 16,729 / test 2,953)|Accuracy 94.6% contro 50.8% della baseline<br>"sempre positive"; F1 0.946 (sulla classe<br>positive); confusion matrix [[1401, 51], [109,<br>1392]]|
|17–18 ott|Slide del mid-term, discorso, guida per le<br>domande tecniche|Vedi la sezione File e link|
|ott|Esplorazione dei due dataset, scelta<br>dell'idea, nome Roomor|Vedi le sezioni sopra|



**Nota sui numeri.** Prima del mid-term Claude aveva fatto dei prototipi per verificare la fattibilità (per esempio Model 1 con micro-F1 circa 0.86 contro 0.80 delle parole chiave). Non sono risultati nostri: valgono solo come riferimento per capire se siamo sulla strada giusta. Anche il controllo dei pesi del Model 3 sull'intero 515K (Staff −1.08, Room −1.07) l'ha fatto Claude, non il nostro codice. 

### **Aggiornamento 23 novembre, sera: cosa va su GitHub.**

- **Su GitHub:** `data/README.md`, `data/Hotel_Database.xlsx` (catalogo dei 40 hotel) e i modelli addestrati in `models/` (`topic_model.joblib` 1.2 MB, `sentiment_model.joblib` 0.2 MB, `impact_weights.csv`, `impact_intercept.json`). Report e demo funzionano subito dopo il clone, senza riaddestrare; i file `.joblib` vanno caricati con scikit-learn 1.8.0.
- **Fuori da GitHub, da Kaggle:** `HRAST.csv`, `Hotel_Reviews.csv` (238 MB, oltre il limite di 100 MB di GitHub).
- **Fuori da GitHub, creati dagli script:** `review_topics.csv` (77 MB), `hrast_train/val/test.csv`, `manual_check.xlsx`.
- **Fuori da GitHub, con recensioni copiate a mano (si condividono su Drive):** `citizenm_zurich_booking.csv`, `google_check.xlsx`, `manual_check.xlsx`, `manual_check_CLAUDE_labels.xlsx`, `WWRSF_HotelReviewRegression.xlsx`. Attenzione: anche i file del controllo a mano contengono i testi delle recensioni, per questo restano fuori.
- Aggiornati `.gitignore`, `data/README.md`, `models/README.md`, `README.md` e `docs/CODE_EXPLANATION.md`.

## **Decisioni prese e perché** 

|Decisione|Perché|
|---|---|
|Sentiment binario (positivo / negativo), senza<br>la classe neutral|Le neutral sono il 3% e mescolano frasi fattuali e<br>frasi con giudizi opposti|
|Model 2 allenato solo su frasi semplici (Aspect<br>= 0)|Le frasi miste vengono spezzate da una regola fissa<br>prima del modello; le metriche si riportano separate<br>per frasi semplici e miste|
|Regressione logistica su TF-IDF per Model 1 e 2|Ogni decisione si spiega parola per parola, dà una<br>probabilità(soglie,casi incerti messi daparte),si|



|Decisione|Perché|
|---|---|
||allena in pochi secondi; slide "Choosing a method"<br>della Week 3|
|Regressione lineare multipla per il Model 3|Il prodotto sono i coefficienti, leggibili da un<br>manager: punti di voto per tema|
|Temi criticati e lodati come colonne separate<br>nel Model 3|Alcuni temi pesano soprattutto quando vanno male<br>(pulizia), altri quando vanno bene (vista): il report<br>dice cosa sistemare e cosa proteggere|
|Una baseline per ogni modello|Week 4: un modello che non batte un'alternativa<br>gratuita non è un prodotto. Temi: parole chiave<br>scritte a mano; sentiment: sempre "positive" e<br>parole chiave; impatto: voto medio|
|Cross-validation a 5 fold per il Model 3 sulle<br>300 righe|Con poche righe un solo split train / test lascerebbe<br>60 righe di test|
|Train / validation / test (70 / 15 / 15, seed 42)<br>da qui in poi|Serve il validation appena si sceglie qualcosa: soglie<br>per tema, regolarizzazione, regressione logistica o<br>DistilBERT|
|DistilBERT solo se batte la regressione logistica<br>sul test|È meno spiegabile e più costoso: deve guadagnarsi il<br>posto|
|Niente scraping|Termini di Booking.com, sezione A15.2|
|Dataset Kaggle mai su GitHub|Licenze e dimensione dei file; si scaricano da Kaggle<br>seguendo il README|
|Su GitHub i file piccoli del team senza testi di recensioni e i modelli addestrati (23 nov)|Filippo deve avere i file che crea Flavio; i modelli (1.4 MB in tutto) servono a report e demo senza riaddestrare|
|Recensioni copiate a mano mai su GitHub (23 nov)|Sono testi scritti dagli ospiti, pubblicati su Booking / Google alle loro condizioni: rimetterli in una repo è una ripubblicazione che non serve al progetto (bastano numeri e grafici); si condividono tra noi su Drive|
|Soglie della regola fissa 0.50 / 0.50, nessun pezzo messo da parte (23 nov)|Scelte su 150 hotel di train con una regola decisa prima: mettere da parte i pezzi incerti perdeva più critiche di quante ne salvasse (sovrapposizione Fix first 0.817 → 0.850)|
|Il voto della singola recensione non si mostra al direttore (23 nov)|Su citizenM il Model 3 per recensione fa peggio della baseline (MAE 0.64 contro 0.58); il voto aggregato dell'hotel invece è accurato (8.17 contro 8.33)|



## **Cosa stiamo facendo ora e cosa resta** 

Il Model 1 (temi) è fatto e salvato in <mark>models/topic_model.joblib</mark> , con la funzione <mark>predict_topics</mark> . Scoperta utile: in HRAST il tema **Generic** riguarda l'edificio e l'hotel nel suo insieme (building, design, decor, entrance, atmosphere), non i giudizi vaghi. Il Model 1 è stato applicato a tutto il 515K <mark>(src/apply_topics.py</mark> ) e il Model 3 è addestrato su tutte le recensioni ( <mark>src/impact_full.py</mark> ). Bootstrap, confronto e analisi di sensibilità sono fatti (22 novembre). Il generatore di report esiste (`src/pipeline.py` e `src/report.py`, nella branch `main`) ed è provato su 3 hotel e verificato sui 224 hotel di test (22 novembre); DistilBERT sul Model 1 batte il TF-IDF su HRAST (22 novembre); la stabilità è confermata con tre seed e si tiene il seed 42; ora si controlla l'errore di conteggio, poi si decide se applicarlo al 515K. Il 23 novembre (branch `flavio-pre-web-development`) è stata costruita la regola fissa: il report funziona anche con recensioni scritte come testo unico ed è stato provato su un hotel svizzero mai visto (citizenM Zürich). Il piano completo, con le caselle da spuntare, è nella roadmap. 

1. **Cuore ML (16–22 nov):** Model 1 con soglia per tema scelta sul validation; Model 2 rifatto con train / validation / test e F1 anche sulle lamentele; regola di divisione delle frasi (**fatta il 23 nov**, valutata contro i campi Booking del 515K e su citizenM Zürich). 

2. **Model 3 e report (23–29 nov):** Model 1 applicato a tutto il 515K; Model 3 su tutte le recensioni con temi criticati e lodati; test su hotel mai visti (divisione per hotel); generatore del report "cosa sistemare per primo". 

3. **Validazione (fino al 6 dic):** test set svizzero, DistilBERT sul sentiment, confronto con un LLM, feedback di utenti veri. 

4. **Consegne (dal 30 nov al pitch):** demo dal vivo (per esempio Streamlit), codice con README, sito web, sezione Trustworthy AI, pitch finale di circa 10 minuti. 

Se manca tempo si taglia in quest'ordine: video, confronto con l'LLM, DistilBERT sui temi, Swiss test set ridotto. Non si tagliano mai Model 1, Model 3 completo, report con demo, README, sito e Trustworthy AI. 

## **Limiti noti e domande aperte** 

### **Limiti già dichiarati al mid-term** 

- **Frasi miste che la regola non può spezzare:** "Clean room with a terrible view" non ha "but" né virgole, quindi riceve un solo sentiment per due temi. 

- **TF-IDF ignora l'ordine delle parole:** "Not as clean as we expected" contiene comunque "clean". 

- • **Non testato su hotel nuovi:** HRAST non dice da quale hotel viene una frase, quindi il 94.6% è probabilmente ottimistico. 

### **Altri limiti da tenere a mente** 

- Il Model 3 misura un' **associazione, non una causa** : chi critica lo staff dà in media un punto in meno, ma sistemare lo staff non garantisce un punto in più. 

- I dati non somigliano ai clienti: 515K è solo inglese, sei grandi città, 2015–2017, quasi metà recensori britannici; i clienti sono hotel svizzeri con recensioni in più lingue. 

- HRAST ha licenza non commerciale: un prodotto vero avrebbe bisogno di etichette proprie. 

- **Lodi lette come critiche (regola fissa, 23 nov):** su citizenM circa 21% dei pezzi mandati al lato negativo erano lodi o fatti neutri; nessuna critica persa. Un tema può essere gonfiato (Noise con "didn't hear a thing"): gli esempi del report vanno riletti. Da dichiarare in Trustworthy AI.

- **Servono punteggiatura, inglese e abbastanza recensioni (23 nov):** senza punteggiatura la regola aiuta poco; le recensioni non in inglese sono ignorate senza avviso; con 30 recensioni il problema n. 1 coincide con quello del report completo solo nel 53% dei casi (80% con 200).

- **Problemi senza un tema:** caldo e aria condizionata (4 recensioni di citizenM) non hanno un tema tra i 21 e finiscono sparsi in Facilities e Comfort.

- **Critiche contate per errore:** nel campo negativo si conta il tema, non il giudizio; circa 30% dei conteggi sbagliati su 50 recensioni di due hotel estremi (12% Savoy, 47% Casa Mimosa), da ricontrollare a mano. Da dichiarare in Trustworthy AI. 

### **Domande aperte** 

- ☐ Ricontrollare a mano, noi due, le etichette di `data/count_check_labels.csv` (errore di conteggio). 

- ☐ Rivedere le etichette di Claude del controllo a mano (`data/manual_check_CLAUDE_labels.xlsx`, colonna `REVIEWED BY`) e aggiornare `results/manual_check_summary.md` con chi le ha riviste. 


- ☑ Date: consegna di tutto (codice, demo, sito, slide, extra) entro l'8 dicembre; presentazione del Gruppo 1 il 15 dicembre. 

- ☑ Feedback del prof sul mid-term (mail del 16 novembre): risultati come grafico o tabella in una pagina web, non screenshot del terminale; analisi dei concorrenti (domanda ancora aperta); una conversazione vera con un direttore d'hotel; demo dal vivo; una slide Trustworthy AI su "cosa succede se lo strumento sbaglia e il direttore ristruttura la cosa sbagliata". 

- ☐ Il prof approva la copia manuale di recensioni per il test set svizzero? 

- ☐ Risposta del contatto nel mondo hotel management. 

## **Come lavoriamo (istruzioni per Claude)** 

- **Chi siamo:** due studenti del major di management, corso introduttivo. **Non sappiamo programmare.** 

- **Codice:** lo costruiamo noi, un passo alla volta, con Claude Code. Claude in chat non ci dà soluzioni già pronte ma **prompt da dare a Claude Code** , divisi in passi, ognuno con una richiesta di spiegazione riga per riga e dei numeri di controllo per sapere se il risultato è giusto. 

- **Lingua:** spiegazioni in italiano semplice, senza dare per scontati concetti di programmazione. Slide, sito, demo e risposte al prof in inglese. 

- **Onestà:** non inventare numeri o fonti. Distinguere sempre i numeri usciti dal nostro codice da quelli dei prototipi di Claude. 

- **Collegamenti al corso:** quando possibile, collegare le scelte alle lezioni (Week 2: train / validation / test; Week 3: classificazione, regressione, "Choosing a method"; Week 4: metriche e baseline; Week 5: pre-training e fine-tuning; Week 6: reti neurali). 

- **Stile grafico:** minimal da startup, marrone opaco (#7a5c46 su fondo crema #f6f2ec, testo #2b2420), font DM Sans e DM Mono. Da evitare tutto ciò che sembra "fatto dall'AI": linee sottili sopra i paragrafi, tre colonne identiche. 

- **Testi per persone esterne** (albergatori, contatti): niente tecnicismi, tono informale, domande brevi. 

## **File e link** 

|Cosa|Dove|
|---|---|
|Questo file di contesto|`docs/internal/` nella repository (branch `main`)|
|To-Do list in ordine di fase (aggiornata da Claude)|`docs/internal/To-Do-List.md`|
|Guida al codice riga per riga (inglese)|`docs/CODE_EXPLANATION.md`|
|Roadmap fino al pitch finale|documento|
|Guida per le domande tecniche (italiano)|documento|
|Guida per le domande tecniche (inglese)|Roomor_technical_QA_study_guide.docx|
|Discorso del mid-term (circa 7:50)|Roomor_midterm_script.docx|
|Slide del mid-term, versione finale|iMLDL_mid-term_group1.pdf(10 slide) e<br>iMLDL_mid-term.pptx|
|Bozza delle slide (prima versione)|deck|
|Documento di progetto iniziale (ottobre)|Progetto_recensioni_hotel.docx|
|Catalogo dei 40 hotel svizzeri (candidati per il test set)|`data/Hotel_Database.xlsx` (su GitHub)|
|Campione da 300 recensioni per il Model 3|WWRSF_HotelReviewRegression.xlsx, foglio<br>"Recensioni"|
|Script di prova di Claude (solo riferimento)|baseline.py,<br>analisi_frasi_miste.py|
|Report punti di forza e debolezza (23 nov)|https://claude.ai/artifact/HCm1oZDk4THn9txEiniU4o|
|Script del 23 novembre (branch `flavio-pre-web-development`)|`src/verify_full_texts.py`, `src/check_booking_hotel.py`, `src/check_pieces.py`, `src/explain_hotel.py`, `experiments/choose_thresholds.py`; risultati in `results/full_texts_check.csv`, `results/manual_check_summary.md`, `results/citizenm_zurich_points.png`|
|Script nostri|Repository GitHub condivisa, branch `main`: pipeline finale in `src/` (topic_model.py, sentiment_model.py, pipeline.py, apply_topics.py, impact_full.py, report.py, verify_hotels.py); versioni precedenti e analisi di sensibilità in `experiments/` (impact_model.py, impact_nothing.py)|
|Dataset|HRAST e 515K da Kaggle (link nella sezione I dati),<br>nella cartella<br>data/ esclusa da GitHub (vedi data/README.md)|
|Materiale del corso|iMLDL-SemesterProject.pdf(brief),<br>5-Preparing-<br>Mid-Term-Talk.pdf, slide delle lezioni M01–M05|



