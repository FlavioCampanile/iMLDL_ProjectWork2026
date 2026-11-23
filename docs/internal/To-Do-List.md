# Roomor · To-Do list

Aggiornata il 23 nov 2026. Claude la aggiorna a ogni passo del progetto, insieme a "Roomor · Contesto del progetto.md".
Si lavora nell'ordine delle fasi: una fase si apre quando la precedente è chiusa.

**Date fisse:** **tutto va consegnato entro l'8 dicembre** (codice, demo, sito, slide finali ed extra). Il Gruppo 1 presenta il **15 dicembre**: tra le due date restano solo le prove del pitch e della demo.

---

## Fase 1 · Chiudere e rifinire il codice

Obiettivo: poter dire che il codice è completo ("runs from the README; honest train, validation, test").

- [ ] **DistilBERT:** decidere se integrarlo nella pipeline (Model 1) o dichiararlo come esperimento non usato nel prodotto
  - [ ] se si integra: provare il secondo criterio (battere il MAE 0.97 su hotel mai visti), misurare il tempo, collegarlo a `report.py`, aggiungere torch / transformers ai requisiti
  - [ ] se non si integra: scriverne il motivo (in `models/README.md` e nel contesto)
- [ ] **Model 2 (sentiment):** rifarlo con train / validation / test (oggi solo train / test 85 / 15), oppure scrivere perché non serve (nessuna scelta fatta sul test)
- [ ] **Rivedere a mano le etichette messe da Claude:** `data/count_check_labels.csv` (errore di conteggio, circa 30%) e `data/manual_check_CLAUDE_labels.xlsx` (colonna `REVIEWED BY`), poi aggiornare `results/manual_check_summary.md`
- [ ] `requirements.txt` completo per tutto il codice della pipeline
- [ ] Prova da zero: cartella pulita, si eseguono gli script nell'ordine del README, i numeri tornano uguali

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
