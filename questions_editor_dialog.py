# In un nuovo file questions_editor_dialog.py o nello stesso, dipende dalla complessità
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLineEdit, QMessageBox
from PyQt6.QtCore import Qt

class QuestionsEditorDialog(QDialog):
    def __init__(self, initial_questions: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Modifica Domande")
        self.setGeometry(200, 200, 600, 400) # Posizione e dimensione della nuova finestra

        self.questions = initial_questions[:] # Crea una copia delle domande iniziali
        self.updated_questions = initial_questions[:] # Questa lista verrà modificata e restituita

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout()

        # Widget per visualizzare le domande
        self.question_list_widget = QListWidget()
        self.question_list_widget.addItems(self.questions)
        
        # Abilita l'editing diretto degli elementi (opzionale)
        self.question_list_widget.setEditTriggers(QListWidget.EditTrigger.DoubleClicked | QListWidget.EditTrigger.SelectedClicked)
        # Quando un elemento viene modificato, aggiorna la lista interna
        self.question_list_widget.itemChanged.connect(self._on_item_changed)


        main_layout.addWidget(self.question_list_widget)

        # Campo per aggiungere una nuova domanda
        add_layout = QHBoxLayout()
        self.new_question_input = QLineEdit()
        self.new_question_input.setPlaceholderText("Digita una nuova domanda qui...")
        add_layout.addWidget(self.new_question_input)

        self.add_button = QPushButton("Aggiungi")
        self.add_button.clicked.connect(self._add_question)
        add_layout.addWidget(self.add_button)
        main_layout.addLayout(add_layout)

        # Bottoni per rimuovere
        remove_layout = QHBoxLayout()
        self.remove_selected_button = QPushButton("Rimuovi Selezionate")
        self.remove_selected_button.clicked.connect(self._remove_selected_questions)
        remove_layout.addWidget(self.remove_selected_button)
        main_layout.addLayout(remove_layout)


        # Bottoni di azione (Salva/Annulla)
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Salva")
        self.save_button.clicked.connect(self.accept) # Chiude la finestra con QDialog.Accepted
        button_layout.addWidget(self.save_button)

        self.cancel_button = QPushButton("Annulla")
        self.cancel_button.clicked.connect(self.reject) # Chiude la finestra con QDialog.Rejected
        button_layout.addWidget(self.cancel_button)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def _on_item_changed(self, item):
        # Aggiorna la lista interna quando un elemento viene modificato direttamente nella QListWidget
        row = self.question_list_widget.row(item)
        if 0 <= row < len(self.updated_questions):
            self.updated_questions[row] = item.text()

    def _add_question(self):
        new_q = self.new_question_input.text().strip()
        if new_q:
            self.question_list_widget.addItem(new_q)
            self.updated_questions.append(new_q) # Aggiunge alla lista interna
            self.new_question_input.clear()
        else:
            QMessageBox.warning(self, "Attenzione", "La domanda non può essere vuota.")

    def _remove_selected_questions(self):
        selected_items = self.question_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Informazione", "Seleziona almeno una domanda da rimuovere.")
            return

        reply = QMessageBox.question(self, "Conferma Rimozione",
                                     f"Sei sicuro di voler rimuovere {len(selected_items)} domande selezionate?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            # Rimuovi gli elementi selezionati dal QListWidget in ordine inverso per evitare problemi di indice
            for item in reversed(self.question_list_widget.selectedItems()):
                row = self.question_list_widget.row(item)
                self.question_list_widget.takeItem(row)
                # Rimuovi dalla lista interna di lavoro
                if 0 <= row < len(self.updated_questions):
                    del self.updated_questions[row]
            # Assicurati che updated_questions sia coerente con l'UI dopo le rimozioni
            self.updated_questions = [self.question_list_widget.item(i).text() for i in range(self.question_list_widget.count())]


    def get_updated_questions(self) -> list:
        # Questo metodo viene chiamato dalla finestra principale per ottenere le domande aggiornate
        return self.updated_questions