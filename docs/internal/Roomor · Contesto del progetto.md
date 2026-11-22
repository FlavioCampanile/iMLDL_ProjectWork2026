# **Roomor · Contesto del progetto** 

Nov 17, 2026 · @Filippo · aggiornato il 22 nov 2026 (struttura della repo)

Questo documento riassume tutto il progetto Roomor: cosa abbiamo fatto, cosa stiamo facendo e come lavoriamo. Per riprendere il lavoro in una chat nuova, incolla il link a questo documento (oppure esportalo in Markdown e allegalo) e scrivi "leggi questo contesto prima di rispondere". Va aggiornato a ogni passo importante. 

## **Il corso e le regole** 

Roomor è il progetto semestrale del corso "Introduction to Machine Learning and Deep Learning" (iMLDL) all'Università di San Gallo (HSG), tenuto dal Prof. Bruno Rodrigues. Il compito è fondare una startup fittizia con un prodotto basato su machine learning. 

- **Team:** Filippo Strafile e Flavio Massimo Campanile. Entrambi devono capire tutto il progetto e parlare in tutte le presentazioni. 

- **Voto del progetto (60% del corso):** mid-term 15% (fatto il 20 ottobre), pitch finale 20%, codice 15%, sito web 10%; extra come bonus (video, demo interattiva, feedback utenti). L'esame scritto vale l'altro 40%. 

- **Scadenze:** feature freeze in settimana 11, pitch finale con demo dal vivo in settimana 12 (circa 10 minuti). Data esatta da confermare. 

- **Obbligatorio:** una sezione "Trustworthy AI" con limiti, rischi e usi impropri, bias dei dati. 

- **Uso dell'AI:** permesso per scrivere codice, ma ognuno di noi deve saper spiegare qualsiasi riga dal vivo. 

- **Dati:** pubblici, con licenza o nostri; vanno citati. Niente scraping di Booking: i suoi termini (sezione A15.2) vietano la raccolta automatica. 

## **L'idea: Roomor** 

Roomor legge le recensioni di un hotel e dice al direttore cosa sistemare per primo, ordinando i problemi per quanto costano in voto degli ospiti. Slogan: _"Your guests talk. We translate."_ 

- **Problema:** un hotel riceve centinaia di recensioni su più piattaforme e in più lingue. Il voto medio (8.4/10 nel 515K) dice quanto sono soddisfatti gli ospiti, non quale parte dell'hotel li delude. Conseguenza: il budget per ristrutturazioni e investimenti rischia di andare nel posto sbagliato. 

- **Utente:** direttori generali e proprietari di hotel indipendenti e piccoli gruppi, senza un team interno di analisi. Partiamo dalla Svizzera. Nello stesso hotel la stessa persona legge le recensioni, decide il budget e firma il contratto. 

- **Cosa entra:** il testo delle recensioni dell'hotel. Nessuna installazione, i modelli non si riaddestrano per ogni hotel. 

- **Cosa esce:** un report con i temi ordinati per impatto. Per ogni tema: quanto spesso gli ospiti se ne lamentano, quanti punti di voto costa in media una critica (e quanti ne aggiunge un 

complimento), e 2–3 frasi vere come esempio. Priorità di un tema = quota di recensioni che lo criticano × peso del tema. 

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
|Regola fissa|Spezza le recensioni<br>su but, however,<br>although… e sulle<br>virgole se un pezzo<br>ha due temi; mette<br>da parte i pezzi con<br>sentiment incerto<br>(probabilità 0.35–<br>0.65); nel 515K<br>finestre di circa 12|Regola scritta a<br>mano, non ML|—|Accuratezza<br>sulle frasi<br>miste|Da costruire|



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
|Etichette|21 temi 0/1; sentiment positive 11,819<br>(51%) / negative 10,504 (45%) /<br>neutral 794 (3%); Aspect = 1 in 2,884<br>frasi miste|Voto 2.5–10 (media 8.4, mediana 8.8, 48% ≥<br>9); testo diviso in Negative_Review e<br>Positive_Review; nessuna etichetta di tema|
|Serve a|Model 1 (temi) e Model 2 (sentiment)|Model 3 (impatto sul voto)|
|Problemi|BOM nel file (<br>encoding="utf-8-sig"); a<br>capo dentro le virgolette (Excel spezza<br>le righe); 8 duplicati; 4 righe con due<br>sentiment; "σ" nella colonna Location<br>(id 4634); nessuna colonna hotel o<br>città|526 duplicati; testo senza punteggiatura;<br>"nessuna critica" scritto in molti modi ("No<br>Negative" 127,890 volte, "Nothing", "N A"…,<br>circa 30%); lat/lng "NA" in 3,268 righe; date<br>mese/giorno/anno; 48% dei recensori dal<br>Regno Unito|



Temi più frequenti in HRAST: Room 5,004, Location 4,729, Staff 3,804. Nel 515K l'hotel si identifica con <mark>Hotel_Address</mark> (1,493 indirizzi per 1,492 nomi). 

**Campione da 300 recensioni (file** **<mark>WWRSF_HotelReviewRegression.xlsx</mark> ).** 300 recensioni vere estratte a caso dal 515K (seed 42) tra le 474,087 che citano almeno uno di 5 temi: Staff, Room, Noise, Location, Restaurant. Le colonne <mark>Neg_<tema></mark> e <mark>Pos_<tema></mark> valgono 1 se nel testo negativo o positivo compare una parola chiave del tema. **I temi sono trovati con parole chiave, non a mano** : va sempre detto così. Le parole chiave sbagliano ("no towels left after service" contato come critica allo Staff). 

**Test set svizzero:** da costruire. Pilota di circa 10 hotel × 30 recensioni in inglese, etichettate a mano da noi due; serve l'approvazione del prof per la copia manuale. 

## **Cosa abbiamo fatto finora** 

Al mid-term del 20 ottobre avevamo il sentiment (Model 2) e una prima versione del Model 3 su 300 recensioni. Il 17 novembre abbiamo costruito il Model 1 (temi): ora i tre modelli esistono tutti, manca il Model 3 su tutto il 515K. 

### **Aggiornamento 17 novembre, pomeriggio.** 

- **Model 1 applicato al 515K** <mark>(apply_topics.py</mark> → <mark>data/review_topics.csv</mark> , 515,212 recensioni dopo i duplicati su tutte le colonne). Testi divisi in finestre di 12 parole con passo 6. Segnaposti di "nessuna critica" svuotati con due regole: testi identici a una lista ("No Negative", "nothing", "n a", "leaving"…) ed espressioni brevi ("can t fault", "no complaints"…) solo in testi di al massimo 8 parole senza but / except / apart / only / however / although / besides. 

- **Filtro con il sentiment provato e scartato.** L'idea era scartare le finestre del campo negativo che il Model 2 giudica lodi. Su un controllo a mano scartava più critiche vere che lodi (Location: 13 sbagliate su 20), perché il Model 2 ha imparato parole di tema come "location" e "clean" come positive, le finestre tagliano le negazioni e il modello non conosce il neutro. Lezione: il 94.6% di 

HRAST non vale su dati diversi. Una regola mirata su "nothing…" è tenuta disattivata, per un'analisi di sensibilità. 

• **Model 3 su tutto il 515K** <mark>(impact_full.py</mark> ): divisione per hotel (1,269 hotel / 432,531 recensioni in train, 224 hotel / 82,681 recensioni in test, nessun hotel in comune). Su hotel mai visti: MAE 0.97 contro 1.29 della baseline (voto medio del train, 8.37), R² 0.36 contro −0.009. Intercetta 8.34. Pesi principali: critiche a Staff −0.80, Clean −0.70, Room −0.61, Bed −0.50, Bathroom −0.46, Noise −0.42; lodi a Staff +0.68, Generic +0.47, View +0.28. Asimmetria tipo Kano: Clean −0.70 / +0.08, Bathroom −0.46 / +0.10, Value for money −0.39 / +0.06; View −0.01 / +0.28. Location lodata nel 48.5% delle recensioni ma vale solo +0.17. Lodi a Lift, Reception e Wi-Fi con segno negativo: temi rari, da verificare con il bootstrap. 

### **Aggiornamento 22 novembre.** 

- **Bootstrap per hotel** (200 ricampionamenti dei 1,269 hotel di train, circa 8 minuti): 37 pesi su 42 affidabili. Non affidabili: Neg_Beach, Neg_View, Pos_Parking, Pos_Beach, e Neg_Bar al limite (con 1,000 ripetizioni l'intervallo tocca lo zero). Pesi principali molto precisi: Neg_Staff da −0.83 a −0.77, Pos_Staff da +0.66 a +0.70. Ricampionare per hotel invece che per recensione rende gli intervalli in media 1.66 volte più larghi (3.4 volte per Pos_Staff): le recensioni dello stesso hotel non sono indipendenti. Pos_Lift (−0.21), Pos_Reception (−0.07) e Pos_Wi-Fi (−0.03) sono affidabili ma con segno controintuitivo: nel report non vanno mostrati tra le cose da proteggere. 

- **Pesi salvati** in <mark>models/impact_weights.csv</mark> (colonne topic, coef, lower, upper, share_pct; share_pct calcolata sul train). 

- **Confronto con le 300 recensioni:** segno uguale in 10 pesi su 10, 9 su 10 dentro l'intervallo della prova; nel modello completo i pesi sono quasi tutti più vicini a zero (Neg_Room da −1.10 a −0.61), perché con 42 colonne l'effetto della camera si divide tra Room, Bed, Bathroom, Clean e Comfort, e perché gli errori del Model 1 attenuano i pesi. 

- **Analisi di sensibilità sulla regola "nothing"** <mark>(impact_nothing.py</mark> ): la regola cambia 582 recensioni su 515,212 (0.11%, voto medio 9.63); differenza massima tra i pesi 0.006 (Neg_Generic, da −0.453 a −0.460); MAE 0.9685 contro 0.9681. Decisione: si tiene la tabella senza regola. 

- **Decisione:** DistilBERT dopo il report, con un limite di 2 giorni, sul Model 1 (temi) perché è quello che costruisce la tabella del Model 3. Successo = batte micro F1 0.857 / macro 0.852 su HRAST e poi il MAE 0.97 su hotel mai visti. 

### **Aggiornamento 22 novembre, struttura della repo (branch `submission`).** 

- **Nuova branch `submission`**, copia di `topic-model`, organizzata già per la consegna. Da qui in poi si lavora su questa branch. **Il codice non è stato toccato**: i file `.py` sono stati solo spostati (git li vede come rinomine al 100%, nessuna riga cambiata). 

- **Nuova struttura:** 
  - `src/` · la pipeline finale: `topic_model.py`, `sentiment_model.py`, `apply_topics.py`, `impact_full.py`. Qui vanno anche i prossimi `pipeline.py` e `report.py`. 
  - `experiments/` · versioni precedenti e analisi di sensibilità, fuori dalla pipeline finale: `impact_model.py` (Model 3 sulle 300 recensioni, mid-term) e `impact_nothing.py` (regola "nothing"). 
  - `app/` · la demo dal vivo (per ora solo un README). 
  - `docs/` · `CODE_EXPLANATION.md`, `TRUSTWORTHY_AI.md` (bozza con i titoli: limiti, rischi, usi impropri, bias dei dati) e `docs/internal/` con questo file e `To-Do-List.md`. 
  - `Data/` e `models/` · restano fuori da GitHub, ma ora ognuna ha un `README.md` tracciato che dice quali file metterci e da quale script nascono. 
  - Nella radice: `README.md` riscritto in inglese per la consegna (nome Roomor, struttura, pipeline, come riprodurre i risultati, dati e licenze), `requirements.txt` con le versioni usate, `.gitignore` aggiornato. 

- **Come si lanciano ora gli script:** sempre dalla cartella principale della repo, non da dentro `src/`: per esempio `python src/topic_model.py`. Funziona senza cambiare il codice perché gli script cercano `Data/` e `models/` a partire dalla cartella da cui si lancia il comando. 

- **Da completare nella struttura:** i link Kaggle in `Data/README.md`, il testo di `docs/TRUSTWORTHY_AI.md`, la demo in `app/`. Il sito web non ha ancora una cartella: da decidere dove ospitarlo. 

|Quando|Cosa|Risultato|
|---|---|---|
|17 nov|Model 1 · temi su HRAST<br>(<br>topic_model.py): 23,105 frasi dopo i<br>duplicati, divise 16,173 / 3,466 / 3,466<br>(train / validation / test, seed 42); 21<br>regressioni logistiche su TF-IDF; soglia per<br>tema scelta sul validation (quasi tutte tra<br>0.05 e 0.30)|Micro F1 0.857 / macro F1 0.852 sul test.<br>Baseline: parole chiave 0.808 / 0.807,<br>"sempre Room" 0.167 / 0.017. Con soglia<br>0.5 il modello era peggio delle parole chiave<br>(0.797 / 0.727). Vince su 15 temi su 21,<br>soprattutto Generic (0.16 → 0.62), Facilities<br>(0.59 → 0.79), Comfort (0.71 → 0.82); le<br>parole chiave vincono su Clean, View,<br>Beach, Bed, Lift, Value for money|



|Quando|Cosa|Risultato|
|---|---|---|
|16 nov|Messaggio a un contatto nel mondo hotel<br>management per un feedback (utilità da<br>1 a 10, come decidono oggi)|In attesa di risposta|
|10 nov|Feedback scritto del prof sul mid-term|Non ancora letto con Claude|
|20 ott|Mid-term talk, 10 slide in PDF, 8 minuti +<br>2 di domande|Fatto; voto non ancora noto|
|19 ott|Model 3 di prova: regressione lineare<br>sulle 300 recensioni, 10 colonne (5 temi<br>criticati / lodati)|MAE 1.01 contro 1.28 della baseline (voto<br>medio), R² 0.33, cross-validation a 5 fold|
|19 ott|Bootstrap del Model 3 (1,000<br>ricampionamenti delle 300 righe)|Affidabili: critiche a Room −1.10, Staff −1.01,<br>Location −0.88; lodi a Staff +0.89,<br>Restaurant +0.41. Non affidabili: Noise e le<br>altre lodi|
|19 ott|Model 2: TF-IDF + regressione logistica su<br>HRAST (train 16,729 / test 2,953)|Accuracy 94.6% contro 50.8% della baseline<br>"sempre positive"; F1 0.946 (sulla classe<br>positive); confusion matrix [[1401, 51], [109,<br>1392]]|
|17–18 ott|Slide del mid-term, discorso, guida per le<br>domande tecniche|Vedi la sezione File e link|
|ott|Esplorazione dei due dataset, scelta<br>dell'idea, nome Roomor|Vedi le sezioni sopra|



**Nota sui numeri.** Prima del mid-term Claude aveva fatto dei prototipi per verificare la fattibilità (per esempio Model 1 con micro-F1 circa 0.86 contro 0.80 delle parole chiave). Non sono risultati nostri: valgono solo come riferimento per capire se siamo sulla strada giusta. Anche il controllo dei pesi del Model 3 sull'intero 515K (Staff −1.08, Room −1.07) l'ha fatto Claude, non il nostro codice. 

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
|Dataset mai su GitHub|Licenze e dimensione dei file; si scaricano da Kaggle<br>seguendo il README|



## **Cosa stiamo facendo ora e cosa resta** 

Il Model 1 (temi) è fatto e salvato in <mark>models/topic_model.joblib</mark> , con la funzione <mark>predict_topics</mark> . Scoperta utile: in HRAST il tema **Generic** riguarda l'edificio e l'hotel nel suo insieme (building, design, decor, entrance, atmosphere), non i giudizi vaghi. Il Model 1 è stato applicato a tutto il 515K <mark>(apply_topics.py</mark> ) e il Model 3 è addestrato su tutte le recensioni ( <mark>impact_full.py</mark> ). Bootstrap, confronto e analisi di sensibilità sono fatti (22 novembre). Ora stiamo costruendo il generatore di report (`src/pipeline.py` e `src/report.py`, nella branch `submission`), con la verifica sui 224 hotel di test; dopo viene DistilBERT sul Model 1. Il piano completo, con le caselle da spuntare, è nella roadmap. 

1. **Cuore ML (16–22 nov):** Model 1 con soglia per tema scelta sul validation; Model 2 rifatto con train / validation / test e F1 anche sulle lamentele; regola di divisione delle frasi, valutata sulle 2,884 frasi miste. 

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

### **Domande aperte** 

- ☐ Data esatta del pitch finale e del feature freeze. 

- ☐ Cosa dice il feedback scritto del prof sul mid-term (10 novembre). 

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
|Questo file di contesto|`docs/internal/` nella repository (branch `submission`)|
|Guida al codice riga per riga (inglese)|`docs/CODE_EXPLANATION.md`|
|Roadmap fino al pitch finale|documento|
|Guida per le domande tecniche (italiano)|documento|
|Guida per le domande tecniche (inglese)|Roomor_technical_QA_study_guide.docx|
|Discorso del mid-term (circa 7:50)|Roomor_midterm_script.docx|
|Slide del mid-term, versione finale|iMLDL_mid-term_group1.pdf(10 slide) e<br>iMLDL_mid-term.pptx|
|Bozza delle slide (prima versione)|deck|
|Documento di progetto iniziale (ottobre)|Progetto_recensioni_hotel.docx|
|Campione da 300 recensioni per il Model 3|WWRSF_HotelReviewRegression.xlsx, foglio<br>"Recensioni"|
|Script di prova di Claude (solo riferimento)|baseline.py,<br>analisi_frasi_miste.py|
|Script nostri|Repository GitHub condivisa, branch `submission`: pipeline finale in `src/` (topic_model.py, sentiment_model.py, apply_topics.py, impact_full.py); versioni precedenti e analisi di sensibilità in `experiments/` (impact_model.py, impact_nothing.py)|
|Dataset|HRAST e 515K da Kaggle (link nella sezione I dati),<br>nella cartella<br>Data/ esclusa da GitHub (vedi Data/README.md)|
|Materiale del corso|iMLDL-SemesterProject.pdf(brief),<br>5-Preparing-<br>Mid-Term-Talk.pdf, slide delle lezioni M01–M05|



