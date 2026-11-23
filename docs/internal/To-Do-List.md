# Roomor · To-Do list

Aggiornata il 23 nov 2026. Claude la aggiorna a ogni passo del progetto, insieme a "Roomor · Contesto del progetto.md".
Si lavora nell'ordine delle fasi: una fase si apre quando la precedente è chiusa.

**Date fisse:** **tutto va consegnato entro l'8 dicembre** (codice, demo, sito, slide finali ed extra). Il Gruppo 1 presenta il **15 dicembre**: tra le due date restano solo le prove del pitch e della demo.

---

## Fase 1 · Chiudere e rifinire il codice

Obiettivo: poter dire che il codice è completo ("runs from the README; honest train, validation, test").

- [x] **DistilBERT sul Model 1: opzione A, resta un esperimento documentato, NON entra nel prodotto** (23 nov). Motivi nel contesto (sezione "Decisioni"); scritto anche in `models/README.md` e `docs/CODE_EXPLANATION.md`
- [x] **Model 2 (sentiment):** deciso di NON aggiungere il validation, spiegato perché in `sentiment_model.py` e in `CODE_EXPLANATION.md` (23 nov)
- [ ] **Rivedere a mano le etichette messe da Claude:** `data/count_check_labels.csv` (errore di conteggio, circa 30%) e `data/manual_check_CLAUDE_labels.xlsx` (colonna `REVIEWED BY`), poi aggiornare `results/manual_check_summary.md`
- [x] Corretti tre commenti superati in `pipeline.py`, `sentiment_model.py` ed `explain_hotel.py` (nessun effetto sui risultati) (23 nov)
- [x] `requirements.txt` controllato: copre tutte le librerie usate da `src/` ed `experiments/`; corretto il commento di openpyxl (serve anche a `check_pieces.py` ed `explain_hotel.py`) (23 nov). Da aggiungere dopo: le librerie della demo nella fase 3 (torch / transformers solo se si fa DistilBERT sul Model 2 e vince)
- [ ] **(a) Quanto l'errore "tema contato, non giudizio" cambia il report** (circa mezza giornata, dopo la revisione delle etichette): per i due hotel del controllo di conteggio (Savoy, Casa Mimosa) rifare il Fix first togliendo le critiche che le etichette dicono sbagliate e confrontarlo con quello di oggi (stessi temi? stesso ordine? stesso n. 1?). Risultato da scrivere nel contesto, in Trustworthy AI e nel pitch; limite: solo due hotel
- [x] **(b) Confidenza per tema nel report: fatta il 23 nov** in `src/report.py` (colonna "Sound negative", avviso sotto il 50%; report identico a prima per il resto). **Resta da verificarla con le etichette di (a).** Descrizione originale: accanto a ogni tema di Fix first, quante delle recensioni che lo criticano il Model 2 legge davvero come negative (es. "solo 4 su 9 suonano negative → controlla gli esempi"). Non cambia conteggi né classifica (i pesi del Model 3 sono calcolati senza filtro). Verificarla con le etichette di (a): i temi con confidenza bassa devono essere quelli con più critiche sbagliate. Da mostrare nella demo (slide 08: "show the confidence, not just the answer")
- [ ] Verificare la confidenza (b) con le etichette riviste: per Savoy e Casa Mimosa, i temi con molte critiche sbagliate hanno confidenza più bassa? Se non serve, decidere se tenerla, cambiarla o toglierla
- [ ] Prova da zero: cartella pulita, si eseguono gli script nell'ordine del README, i numeri tornano uguali (dopo (a) e (b)). Nota: con la confidenza `verify_full_texts.py` e `choose_thresholds.py` sono un po' più lenti (un passaggio in più dei due modelli per ogni report)

### Opzionali della fase 1 (solo se resta tempo)

- [ ] **DistilBERT sul Model 2 (sentiment)**, circa 1–2 giorni. Colpisce un errore vero del prodotto (lodi o fatti neutri letti come critiche dalla regola fissa, es. "didn't hear a thing") e non obbliga a rifare tabella del 515K e Model 3: il Model 2 serve solo alla regola fissa e agli esempi del report.
  - [ ] Quaderno Colab come quello dei temi, 2 classi (positivo / negativo), stesse frasi del Model 2; validation ricavato dal train (il test non si tocca) per scegliere l'epoca
  - [ ] Regola decisa prima: si tiene solo se batte il TF-IDF (accuracy 0.9458) sul test; prova a parte sulle frasi con negazione
  - [ ] Se vince: rifare le verifiche della regola fissa (`verify_full_texts.py`, `check_booking_hotel.py`, controllo a mano) e misurare il tempo senza GPU (serve per la demo)
  - Non risolve le lodi scritte nel campo "disliked" di Booking (lì il Model 2 non entra); il modello pesa circa 268 MB
- [ ] **Opzione B: DistilBERT sul Model 1, ultima prova per arricchire il pitch** (circa 1 giornata, 3–4 ore a mano). Non cambia il prodotto: serve a dire nel pitch se DistilBERT conterebbe meno critiche sbagliate.
  - [ ] Passo 0: scrivere nel contesto la regola PRIMA di guardare (es. critiche contate per errore giù di almeno 10 punti, nessuna critica vera persa in più)
  - [ ] Passo 1: ritrovare `data/count_check_labels.csv` (non è nel PC di Flavio) e rivedere le etichette (coincide con la revisione obbligatoria qui sopra)
  - [ ] Passo 2: script locale che pulisce le 50 recensioni come il report e le taglia nelle stesse finestre di 12 parole (CSV per Colab)
  - [ ] Passo 3: salvare in un JSON le 21 soglie di DistilBERT seed 42 già scelte sul validation (`compare_distilbert.py`), senza usare il test
  - [ ] Passo 4: quaderno Colab corto: carica il modello seed 42 da Drive (`Roomor/distilbert_topics_8epochs/model`), probabilità per finestra, tempo senza GPU
  - [ ] Passo 5: script di confronto TF-IDF / DistilBERT / etichette, coppia recensione–tema
  - [ ] Passo 6: etichettare a mano le coppie trovate solo da DistilBERT
  - [ ] Passo 7: applicare la regola, scrivere il risultato nel contesto, in `CODE_EXPLANATION.md` (parte 9) e in Trustworthy AI
  - Limiti: 50 recensioni di due hotel estremi; le lodi nel campo "disliked" restano un errore anche con DistilBERT; citizenM non è utilizzabile senza etichette dei temi

## Fase 2 · Test set svizzero (10 hotel × 30 recensioni)

- [x] Catalogo di 40 hotel candidati: `data/Hotel_Database.xlsx` (città, cluster, stelle, ristorante, wellness, parcheggio, sci); elenco completo nel contesto, sezione "I dati" (23 nov)
- [ ] Scegliere i 10 hotel tra i 40: quali località, quante stelle, quali servizi offrono, quanti per ognuna di queste caratteristiche (citizenM Zürich, H001, è già stato usato)
- [ ] Scegliere le recensioni: quante per ogni tipologia di hotel, quali aspetti, quante per ogni aspetto, quante positive / negative / neutre
- [ ] Costruire i due dataset: dataset 1 e dataset 2
- [ ] Etichettare a mano (noi due) e far girare il report; confronto con citizenM Zürich (già fatto, 1 hotel)
- [ ] Riportare i risultati nel contesto

## Fase 3 · Live demo

- [ ] App (per esempio Streamlit): recensioni in ingresso, report "Fix first / Protect" in uscita
- [ ] Mostrare la confidenza, non solo la risposta (Week 4)
- [ ] Risultati come tabelle e grafici, non output del terminale (feedback del prof)
- [ ] Salvare ogni richiesta (primo test set dal mondo reale)
- [ ] Pubblicata online gratis; un estraneo la usa senza che noi tocchiamo la tastiera
- [ ] Librerie della demo in `requirements.txt`; codice in `app/`

## Fase 4 · Sito web

- [ ] Online, con un URL
- [ ] Chi / cosa / perché in 30 secondi
- [ ] Metodologia e limiti, risultati su hotel mai visti con baseline (grafici)
- [ ] Collegato alla demo (se la demo è usabile dal sito conta anche come extra)
- [ ] Grafica curata (stile Roomor: #7a5c46, crema #f6f2ec, DM Sans / DM Mono)

## Fase 5 · Trustworthy AI nella repo

Obbligatorio: senza limiti e rischi il voto del progetto si ferma a 4.0.

- [ ] Scrivere `docs/TRUSTWORTHY_AI.md`: limiti (dove sbaglia e come lo sappiamo), rischi e usi impropri (chi può essere danneggiato), bias e dati (da dove vengono, cosa lasciano fuori)
- [ ] Risposta alla domanda del prof: cosa succede se lo strumento sbaglia e il direttore ristruttura la cosa sbagliata
- [ ] Dichiarazioni: test set di DistilBERT usato quattro volte; etichette pre-messe con un modello AI e riviste da noi

## Fase 6 · README completo

- [ ] Struttura e ordine degli script aggiornati (script del 23 nov, regola fissa, demo, test set svizzero)
- [ ] Link a demo e sito; dati e licenze; sezione Trustworthy AI
- [x] Riscritto `docs/CODE_EXPLANATION.md` su tutto il codice (23 nov)
- [ ] Aggiornare `docs/CODE_EXPLANATION.md` con le modifiche delle fasi 1–3 (DistilBERT, Model 2, test set svizzero, demo)

## Fase 7 · Slide, pitch e giorno della presentazione

- [ ] Analisi dei concorrenti (TrustYou, Customer Alliance, MARA): cosa fanno e dove siamo diversi (domanda aperta del prof)
- [ ] Slide finali (circa 10 minuti + demo): problema, utente, soluzione, come funziona l'AI, risultati su dati mai visti contro baseline, limiti e Trustworthy AI. **Consegna entro l'8 dicembre**
- [ ] Prove del pitch a tempo (al mid-term siamo andati lunghi); parlano tutti e due
- [ ] Prova della demo su un PC pulito e piano B se la rete non va
- [ ] Ripasso: ognuno sa spiegare tutto il progetto e qualsiasi riga di codice

## Fase 8 · Miglioramenti opzionali (bonus, massimo +0.25)

- [ ] Feedback di un professionista del settore alberghiero (contatto del 16 nov in attesa)
- [ ] Video promozionale
- [ ] Altri extra solo se aggiungono valore vero

---

## Fatto

- [x] Model 1, 2 e 3; report; verifica sui 224 hotel; DistilBERT con tre seed (17–22 nov)
- [x] Regola fissa per recensioni scritte come testo unico; test su citizenM Zürich; merge di `flavio-pre-web-development` in `main` (23 nov)
- [x] Su GitHub il catalogo dei 40 hotel e i modelli addestrati; recensioni copiate a mano fuori da GitHub, da passare a Filippo su Drive (23 nov)
