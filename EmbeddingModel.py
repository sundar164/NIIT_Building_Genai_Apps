import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
from datetime import datetime


class BioBERTRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BioBERT RAG Pipeline - Biomedical Q&A System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f4f8')

        # Sample biomedical documents
        self.documents = [
            {
                "title": "COVID-19 Pathophysiology",
                "content": "SARS-CoV-2 is a novel coronavirus that primarily affects the respiratory system. The virus enters cells through the ACE2 receptor. Common symptoms include fever (87.9%), dry cough (67.7%), and fatigue (38.1%). Severe cases may develop acute respiratory distress syndrome (ARDS)."
            },
            {
                "title": "mRNA Vaccine Mechanism",
                "content": "mRNA vaccines contain genetic instructions that teach cells to produce a harmless spike protein found on the virus's surface. The immune system recognizes this protein and builds antibodies. Clinical trials showed 95% efficacy for Pfizer-BioNTech and 94.1% for Moderna vaccines in preventing symptomatic COVID-19."
            },
            {
                "title": "Antiviral Treatment Options",
                "content": "Remdesivir is a nucleotide analog that inhibits viral RNA polymerase. Studies show it reduces recovery time by 31%. Paxlovid (nirmatrelvir/ritonavir) is effective when administered within 5 days of symptom onset, reducing hospitalization risk by 89% in high-risk patients."
            },
            {
                "title": "Long COVID Symptoms",
                "content": "Post-acute sequelae of SARS-CoV-2 infection (PASC) affects approximately 10-30% of infected individuals. Common symptoms include persistent fatigue, brain fog, shortness of breath, and chest pain. Symptoms can last for months after initial infection."
            },
            {
                "title": "Monoclonal Antibody Therapy",
                "content": "Monoclonal antibodies are laboratory-made proteins that mimic the immune system's ability to fight off harmful pathogens. Bamlanivimab and casirivimab/imdevimab have received emergency use authorization. They work best when given early in the disease course."
            }
        ]

        self.current_step = 0
        self.is_processing = False
        self.retrieved_chunks = []
        self.engineered_prompt = ""

        self.setup_ui()

    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f4f8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Title
        title_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        title_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(title_frame, text="BioBERT RAG Pipeline", font=('Arial', 24, 'bold'),
                 bg='white', fg='#1e40af').pack(pady=10)
        tk.Label(title_frame, text="Retrieval-Augmented Generation for Biomedical Q&A",
                 font=('Arial', 12), bg='white', fg='#64748b').pack(pady=(0, 10))

        # Content area
        content_frame = tk.Frame(main_frame, bg='#f0f4f8')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Pipeline steps
        left_panel = tk.Frame(content_frame, bg='white', relief=tk.RAISED, bd=2, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        left_panel.pack_propagate(False)

        tk.Label(left_panel, text="Pipeline Steps", font=('Arial', 14, 'bold'),
                 bg='white', fg='#1e293b').pack(pady=10)

        # Steps frame with scrollbar
        steps_canvas = tk.Canvas(left_panel, bg='white', highlightthickness=0)
        steps_scrollbar = ttk.Scrollbar(left_panel, orient="vertical", command=steps_canvas.yview)
        self.steps_frame = tk.Frame(steps_canvas, bg='white')

        self.steps_frame.bind("<Configure>", lambda e: steps_canvas.configure(scrollregion=steps_canvas.bbox("all")))
        steps_canvas.create_window((0, 0), window=self.steps_frame, anchor="nw")
        steps_canvas.configure(yscrollcommand=steps_scrollbar.set)

        steps_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        steps_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.create_pipeline_steps()

        # Document store
        doc_frame = tk.LabelFrame(left_panel, text="Document Store", font=('Arial', 11, 'bold'),
                                  bg='white', fg='#4338ca', padx=10, pady=10)
        doc_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        doc_text = scrolledtext.ScrolledText(doc_frame, height=10, width=35, wrap=tk.WORD,
                                             font=('Arial', 9), bg='#f8fafc')
        doc_text.pack(fill=tk.BOTH, expand=True)

        for i, doc in enumerate(self.documents, 1):
            doc_text.insert(tk.END, f"{i}. {doc['title']}\n", 'title')
            doc_text.insert(tk.END, f"   {doc['content'][:80]}...\n\n", 'content')

        doc_text.tag_config('title', foreground='#4338ca', font=('Arial', 9, 'bold'))
        doc_text.tag_config('content', foreground='#475569')
        doc_text.config(state=tk.DISABLED)

        # Right panel - Query and results
        right_panel = tk.Frame(content_frame, bg='#f0f4f8')
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Query section
        query_frame = tk.Frame(right_panel, bg='white', relief=tk.RAISED, bd=2)
        query_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(query_frame, text="Query Interface", font=('Arial', 14, 'bold'),
                 bg='white', fg='#1e293b').pack(pady=10, padx=10, anchor='w')

        tk.Label(query_frame, text="Ask a biomedical question:", font=('Arial', 10),
                 bg='white', fg='#475569').pack(pady=(0, 5), padx=10, anchor='w')

        input_frame = tk.Frame(query_frame, bg='white')
        input_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.query_entry = tk.Entry(input_frame, font=('Arial', 11), relief=tk.SOLID, bd=1)
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8)
        self.query_entry.bind('<Return>', lambda e: self.run_rag_pipeline())

        self.search_btn = tk.Button(input_frame, text="Search", font=('Arial', 10, 'bold'),
                                    bg='#4f46e5', fg='white', cursor='hand2',
                                    command=self.run_rag_pipeline, padx=20)
        self.search_btn.pack(side=tk.LEFT, padx=(10, 0))

        # Sample queries
        sample_frame = tk.Frame(query_frame, bg='white')
        sample_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        tk.Label(sample_frame, text="Sample queries:", font=('Arial', 9),
                 bg='white', fg='#64748b').pack(anchor='w')

        samples = [
            "What are the symptoms of COVID-19?",
            "How do mRNA vaccines work?",
            "What treatments are available for COVID-19?"
        ]

        for sample in samples:
            btn = tk.Button(sample_frame, text=sample, font=('Arial', 8),
                            bg='#e0e7ff', fg='#4338ca', relief=tk.FLAT, cursor='hand2',
                            command=lambda s=sample: self.set_query(s))
            btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Results notebook
        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Retrieved Chunks
        chunks_frame = tk.Frame(self.results_notebook, bg='white')
        self.results_notebook.add(chunks_frame, text="Retrieved Chunks")

        tk.Label(chunks_frame, text="Top-k Relevant Chunks from Vector DB",
                 font=('Arial', 12, 'bold'), bg='white', fg='#1e293b').pack(pady=10, padx=10, anchor='w')

        self.chunks_text = scrolledtext.ScrolledText(chunks_frame, wrap=tk.WORD, font=('Arial', 10),
                                                     bg='#fffbeb', relief=tk.SOLID, bd=1)
        self.chunks_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.chunks_text.config(state=tk.DISABLED)

        # Tab 2: Engineered Prompt
        prompt_frame = tk.Frame(self.results_notebook, bg='white')
        self.results_notebook.add(prompt_frame, text="Engineered Prompt ⭐")

        header_frame = tk.Frame(prompt_frame, bg='#f5f3ff')
        header_frame.pack(fill=tk.X, pady=10, padx=10)

        tk.Label(header_frame, text="🔮 Step 7: Prompt Engineering",
                 font=('Arial', 12, 'bold'), bg='#f5f3ff', fg='#6d28d9').pack(pady=5)
        tk.Label(header_frame, text="This is where retrieved context is formatted into an optimized prompt",
                 font=('Arial', 9), bg='#f5f3ff', fg='#7c3aed').pack()

        self.prompt_text = scrolledtext.ScrolledText(prompt_frame, wrap=tk.WORD, font=('Courier', 9),
                                                     bg='#faf5ff', relief=tk.SOLID, bd=1)
        self.prompt_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.prompt_text.config(state=tk.DISABLED)

        # Tab 3: LLM Response
        response_frame = tk.Frame(self.results_notebook, bg='white')
        self.results_notebook.add(response_frame, text="LLM Response")

        tk.Label(response_frame, text="Final Answer",
                 font=('Arial', 12, 'bold'), bg='white', fg='#1e293b').pack(pady=10, padx=10, anchor='w')

        self.response_text = scrolledtext.ScrolledText(response_frame, wrap=tk.WORD, font=('Arial', 11),
                                                       bg='#f0fdf4', relief=tk.SOLID, bd=1)
        self.response_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.response_text.config(state=tk.DISABLED)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN,
                              anchor=tk.W, font=('Arial', 9), bg='#e2e8f0', fg='#1e293b')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_pipeline_steps(self):
        self.step_labels = []
        self.step_indicators = []

        steps = [
            ("1. Import HuggingFace Embeddings", "✓", '#10b981'),
            ("2. Create BioBertEmbeddings Class", "✓", '#10b981'),
            ("3. Download BioBERT Model", "✓", '#10b981'),
            ("4. Create Vector Store Wrapper", "✓", '#10b981'),
            ("5. Initialize Vector DB & Load Docs", "○", '#94a3b8'),
            ("6. Split Documents into Chunks", "○", '#94a3b8'),
            ("7. Perform Similarity Search (MMR)", "○", '#94a3b8'),
            ("8. Prompt Engineering ⭐", "○", '#94a3b8'),
            ("9. Get LLM Response", "○", '#94a3b8')
        ]

        for i, (text, indicator, color) in enumerate(steps):
            frame = tk.Frame(self.steps_frame, bg='white' if i < 4 else '#f8fafc',
                             relief=tk.SOLID, bd=1)
            frame.pack(fill=tk.X, padx=5, pady=3)

            indicator_label = tk.Label(frame, text=indicator, font=('Arial', 12, 'bold'),
                                       fg=color, bg=frame['bg'], width=3)
            indicator_label.pack(side=tk.LEFT, padx=5)

            step_label = tk.Label(frame, text=text, font=('Arial', 10),
                                  fg='#1e293b' if i < 4 else '#64748b',
                                  bg=frame['bg'], anchor='w')
            step_label.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)

            self.step_labels.append(step_label)
            self.step_indicators.append((indicator_label, frame))

    def set_query(self, query):
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, query)

    def update_step(self, step_num, status='processing'):
        if step_num >= len(self.step_indicators):
            return

        indicator_label, frame = self.step_indicators[step_num]

        if status == 'processing':
            indicator_label.config(text='⟳', fg='#3b82f6')
            frame.config(bg='#dbeafe')
            self.step_labels[step_num].config(bg='#dbeafe', fg='#1e40af', font=('Arial', 10, 'bold'))
        elif status == 'complete':
            indicator_label.config(text='✓', fg='#10b981')
            frame.config(bg='#d1fae5')
            self.step_labels[step_num].config(bg='#d1fae5', fg='#065f46')
        elif status == 'highlight':
            indicator_label.config(text='⭐', fg='#7c3aed')
            frame.config(bg='#f5f3ff')
            self.step_labels[step_num].config(bg='#f5f3ff', fg='#6d28d9', font=('Arial', 10, 'bold'))

        self.root.update()

    def run_rag_pipeline(self):
        query = self.query_entry.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a question.")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.search_btn.config(state=tk.DISABLED, text="Processing...")

        # Clear previous results
        self.chunks_text.config(state=tk.NORMAL)
        self.chunks_text.delete(1.0, tk.END)
        self.chunks_text.config(state=tk.DISABLED)

        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.config(state=tk.DISABLED)

        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.config(state=tk.DISABLED)

        # Run pipeline in thread
        thread = threading.Thread(target=self.process_pipeline, args=(query,))
        thread.daemon = True
        thread.start()

    def process_pipeline(self, query):
        try:
            # Step 5: Initialize Vector DB
            self.status_var.set("Step 5: Initializing Vector DB and loading documents...")
            self.update_step(4, 'processing')
            time.sleep(0.8)
            self.update_step(4, 'complete')

            # Step 6: Chunk documents
            self.status_var.set("Step 6: Splitting documents into chunks...")
            self.update_step(5, 'processing')
            time.sleep(0.8)
            self.update_step(5, 'complete')

            # Step 7: Similarity search
            self.status_var.set("Step 7: Performing similarity search (MMR)...")
            self.update_step(6, 'processing')
            time.sleep(1.0)

            # Retrieve top-k chunks (simulate)
            self.retrieved_chunks = self.retrieve_relevant_chunks(query)
            self.display_chunks()
            self.update_step(6, 'complete')

            # Step 8: Prompt Engineering
            self.status_var.set("Step 8: Engineering prompt with retrieved context...")
            self.update_step(7, 'processing')
            time.sleep(1.0)

            self.engineered_prompt = self.build_prompt(query, self.retrieved_chunks)
            self.display_prompt()
            self.update_step(7, 'highlight')

            # Step 9: Get LLM response
            self.status_var.set("Step 9: Generating LLM response...")
            self.update_step(8, 'processing')
            time.sleep(1.2)

            response = self.generate_response(query)
            self.display_response(response)
            self.update_step(8, 'complete')

            self.status_var.set(f"✓ Pipeline complete! ({datetime.now().strftime('%H:%M:%S')})")

        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Pipeline Error", str(e))
        finally:
            self.is_processing = False
            self.search_btn.config(state=tk.NORMAL, text="Search")

    def retrieve_relevant_chunks(self, query):
        # Simple keyword-based retrieval simulation
        query_lower = query.lower()
        scored_docs = []

        for doc in self.documents:
            score = 0
            content_lower = (doc['title'] + ' ' + doc['content']).lower()

            keywords = ['symptom', 'vaccine', 'treatment', 'covid', 'mrna', 'antibod', 'long covid']
            for keyword in keywords:
                if keyword in query_lower and keyword in content_lower:
                    score += 2

            if score > 0 or any(word in content_lower for word in query_lower.split()):
                scored_docs.append((score, doc))

        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:2]]

    def build_prompt(self, query, chunks):
        context = ""
        for i, chunk in enumerate(chunks, 1):
            context += f"[{i}] {chunk['title']}:\n{chunk['content']}\n\n"

        prompt = f"""You are a biomedical expert assistant. Use the following retrieved context to answer the user's question accurately and concisely.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
- Base your answer on the provided context
- Cite specific information from the context when relevant
- If the context doesn't contain enough information, acknowledge this
- Use clear, professional medical terminology
- Keep the response focused and factual

ANSWER:"""

        return prompt

    def generate_response(self, query):
        # Simulate LLM response based on query
        query_lower = query.lower()

        if 'symptom' in query_lower:
            return ("Based on the retrieved medical literature, COVID-19 symptoms typically include "
                    "fever (87.9%), dry cough (67.7%), and fatigue (38.1%). The virus primarily affects "
                    "the respiratory system through the ACE2 receptor. Severe cases may progress to acute "
                    "respiratory distress syndrome (ARDS).")
        elif 'vaccine' in query_lower or 'mrna' in query_lower:
            return ("mRNA vaccines work by delivering genetic instructions that teach cells to produce "
                    "the spike protein found on the virus's surface. The immune system recognizes this protein "
                    "and builds antibodies. Clinical trials demonstrated high efficacy: 95% for Pfizer-BioNTech "
                    "and 94.1% for Moderna in preventing symptomatic COVID-19.")
        elif 'treatment' in query_lower:
            return ("Current treatment options include antiviral medications. Remdesivir, a nucleotide analog "
                    "that inhibits viral RNA polymerase, reduces recovery time by 31%. Paxlovid "
                    "(nirmatrelvir/ritonavir) is most effective when given within 5 days of symptom onset, "
                    "reducing hospitalization risk by 89% in high-risk patients.")
        elif 'long covid' in query_lower or 'post-acute' in query_lower:
            return ("Post-acute sequelae of SARS-CoV-2 infection (PASC), commonly known as long COVID, "
                    "affects approximately 10-30% of infected individuals. Common persistent symptoms include "
                    "fatigue, brain fog, shortness of breath, and chest pain. These symptoms can persist for "
                    "months following the initial infection.")
        else:
            return ("Based on the available medical literature in the database, COVID-19 is caused by "
                    "SARS-CoV-2 and has various manifestations. Multiple treatment approaches exist including "
                    "antiviral medications and vaccines. For specific medical advice, please consult with "
                    "healthcare professionals.")

    def display_chunks(self):
        self.chunks_text.config(state=tk.NORMAL)
        self.chunks_text.delete(1.0, tk.END)

        if not self.retrieved_chunks:
            self.chunks_text.insert(tk.END, "No relevant chunks found.")
        else:
            for i, chunk in enumerate(self.retrieved_chunks, 1):
                self.chunks_text.insert(tk.END, f"Rank {i}: {chunk['title']}\n", 'title')
                self.chunks_text.insert(tk.END, f"{chunk['content']}\n\n", 'content')

        self.chunks_text.tag_config('title', foreground='#b45309', font=('Arial', 10, 'bold'))
        self.chunks_text.tag_config('content', foreground='#78350f')
        self.chunks_text.config(state=tk.DISABLED)
        self.results_notebook.select(0)

    def display_prompt(self):
        self.prompt_text.config(state=tk.NORMAL)
        self.prompt_text.delete(1.0, tk.END)
        self.prompt_text.insert(tk.END, self.engineered_prompt)
        self.prompt_text.config(state=tk.DISABLED)

    def display_response(self, response):
        self.response_text.config(state=tk.NORMAL)
        self.response_text.delete(1.0, tk.END)
        self.response_text.insert(tk.END, response)
        self.response_text.config(state=tk.DISABLED)
        self.results_notebook.select(2)


if __name__ == "__main__":
    root = tk.Tk()
    app = BioBERTRAGApp(root)
    root.mainloop()
