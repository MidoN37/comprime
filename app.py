import streamlit as st
import pandas as pd
import random
import unicodedata
import re
from pathlib import Path

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Pharm Mnemonics Quiz")

# --- Project Setup & File Paths ---
try:
    APP_DIR = Path(__file__).parent
except NameError:
    APP_DIR = Path.cwd() # Fallback for when __file__ is not defined (e.g. in some notebooks)

PROJECT_ROOT = APP_DIR # Assuming your script is in the root or a subdir
MNEMONICS_XLSX_PATH = PROJECT_ROOT / "QUIZ" / "Mnemonics.xlsx" # Corrected to relative path
IMAGE_CSV_PATH = PROJECT_ROOT / "QUIZ" / "github_image_urls_CATEGORIZED.csv" # This was already good

# --- Load Mnemonics Data (New) ---
@st.cache_data(show_spinner="Loading mnemonics data...")
def load_mnemonics_data(path):
    try:
        if not path.exists():
            st.error(f"Mnemonics data XLSX not found: {path}")
            return None
        df = pd.read_excel(path, keep_default_na=False)
        required_mnemonics = ['Commercial Name', 'Indication (French Keyword)', 'Mnemonic']
        if not all(col in df.columns for col in required_mnemonics):
            st.error(f"Mnemonics XLSX missing required columns. Need: {required_mnemonics}. Found: {list(df.columns)}")
            return None
        # Ensure 'Category' exists, even if it's a placeholder for this new quiz type
        if 'Category' not in df.columns:
            df['Category'] = "Mnemonics Quiz" # Or derive from filename/sheet if applicable
        if 'Sheet' not in df.columns: # Sheet might be used for sectioning
            df['Sheet'] = "Full List"
        # Standardize column names if needed to match your existing CSV structure for image lookup
        df.rename(columns={'Commercial Name': 'MedicationName',
                           'Indication (French Keyword)': 'IndicationFrench', # New column for the indication
                           'Mnemonic': 'MnemonicText'}, # New column for mnemonic
                  inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load/parse mnemonics XLSX: {e}")
        return None

# --- Load Image URL Data (Existing - slightly modified for pre-normalization) ---
@st.cache_data(show_spinner="Loading image data...")
def load_image_data(path):
    try:
        if not path.exists():
            st.warning(f"Image data CSV not found: {path}. Images might not load.")
            return pd.DataFrame(columns=['category', 'filename', 'raw_url', '_norm_cat', '_norm_filename']) # Ensure pre-norm cols exist
        try:
            df = pd.read_csv(path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='utf-8-sig')
        except Exception as e:
             st.error(f"Failed to load image CSV with specified encodings: {e}")
             return pd.DataFrame(columns=['category', 'filename', 'raw_url', '_norm_cat', '_norm_filename'])

        required_images = ['category', 'filename', 'raw_url']
        if not all(col in df.columns for col in required_images):
            st.error(f"Image CSV missing required columns. Required: {required_images}. Found: {list(df.columns)}")
            return pd.DataFrame(columns=['category', 'filename', 'raw_url', '_norm_cat', '_norm_filename'])

        # Pre-normalize columns for faster lookup
        df['_norm_cat'] = df['category'].astype(str).apply(normalize_text)
        # Extract filename without extension for matching MedicationName
        df['_norm_filename'] = df['filename'].astype(str).str.extract(r'([^/\\.]+)(?:\.[^.]*$|$)', expand=False).fillna('').apply(normalize_text)
        return df
    except Exception as e:
        st.error(f"Error loading image CSV: {e}")
        return pd.DataFrame(columns=['category', 'filename', 'raw_url', '_norm_cat', '_norm_filename'])

# --- Normalize function for matching (Existing) ---
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text.lower().strip()

# --- Get Image URL (Modified to use MedicationName more directly from mnemonics_df) ---
def get_image_url(df_images, category, medication_name): # Removed sheet, as it might not be relevant for image lookup if category+med_name is unique
    # --- Determine Override URL (Keep your existing logic if needed) ---
    # base_override_url = "https://raw.githubusercontent.com/MidoN37/pharma-data-viewer/refs/heads/master/assets/images"
    # # Example:
    # if normalize_text(category) == "antibiotiques" and normalize_text(medication_name).startswith("amoxi"):
    #     return f"{base_override_url}/Antibiotiques/Amoxicilline_Example.jpg"
    # Add your specific override logic here if any based on category/med_name

    if df_images.empty or not medication_name or not category: return None

    norm_med = normalize_text(medication_name)
    norm_cat = normalize_text(category) # Assuming category is still relevant from df_images

    img_row = df_images[
        (df_images['_norm_cat'] == norm_cat) &
        (df_images['_norm_filename'] == norm_med)
    ]

    if not img_row.empty:
        return img_row.iloc[0]['raw_url']
    # Fallback: try matching just by medication name if category match fails (might be less accurate)
    img_row_med_only = df_images[df_images['_norm_filename'] == norm_med]
    if not img_row_med_only.empty:
        st.warning(f"Image found for '{medication_name}' by name only, category '{category}' might not match image metadata.")
        return img_row_med_only.iloc[0]['raw_url']

    return None


# --- Display Image (Existing) ---
def display_image(url, container):
    if url:
        container.markdown(f'<a href="{url}" target="_blank"><img src="{url}" alt="Medication Image" style="max-width:100%; max-height: 350px; object-fit: contain; margin-top: 10px;"/></a>', unsafe_allow_html=True)

# --- Sectioning Logic ---
NUM_QUIZ_SECTIONS = 15

def get_quiz_sections(df_mnemonics):
    if df_mnemonics is None or df_mnemonics.empty:
        return {}
    
    total_meds = len(df_mnemonics)
    meds_per_section = (total_meds + NUM_QUIZ_SECTIONS - 1) // NUM_QUIZ_SECTIONS # Ceiling division
    
    sections = {}
    for i in range(NUM_QUIZ_SECTIONS):
        start_idx = i * meds_per_section
        end_idx = min((i + 1) * meds_per_section, total_meds)
        if start_idx < end_idx: # Ensure section is not empty
            section_name = f"Section {i+1} (Meds {start_idx+1}-{end_idx})"
            sections[section_name] = df_mnemonics.iloc[start_idx:end_idx].copy()
    return sections

# --- Main Execution ---
if __name__ == "__main__":
    df_mnemonics_all = load_mnemonics_data(MNEMONICS_XLSX_PATH)
    df_images = load_image_data(IMAGE_CSV_PATH)

    if df_mnemonics_all is None:
        st.error("Mnemonics quiz data could not be loaded. Stopping application.")
        st.stop()

    quiz_sections = get_quiz_sections(df_mnemonics_all)

    # Initialize session state keys
    default_session_values = {
        'selected_quiz_section_name': None,
        'question_index': 0,
        'answers': {}, # Stores user's selected option for each question_index
        'feedback_shown': {}, # Stores if feedback (correct/incorrect + mnemonic) has been shown for an index
        'show_result': False,
        'current_quiz_df': pd.DataFrame(),
        'current_question_options': {}, # Stores the shuffled options for each question_index
        'quiz_loaded': False,
        'quiz_type': 'Mnemonics' # Could be used to switch between old/new quiz
    }
    for key, default_value in default_session_values.items():
        st.session_state.setdefault(key, default_value)

    # --- Sidebar ---
    st.sidebar.title("Quiz Settings")

    if not quiz_sections:
        st.sidebar.error("No quiz sections available.")
    else:
        section_names = list(quiz_sections.keys())
        current_selection_section = st.session_state.selected_quiz_section_name
        
        # Ensure current_selection_section is valid, otherwise default to the first
        if current_selection_section not in section_names and section_names:
            current_selection_section = section_names[0]
        
        selected_section_name_sb = st.sidebar.selectbox(
            "Select Quiz Section", options=section_names,
            index=section_names.index(current_selection_section) if current_selection_section in section_names else 0,
            key='sb_section'
        )

        if selected_section_name_sb != st.session_state.selected_quiz_section_name:
            st.session_state.selected_quiz_section_name = selected_section_name_sb
            # Reset quiz state when section changes
            st.session_state.update({
                'current_quiz_df': pd.DataFrame(), 'question_index': 0, 'answers': {},
                'feedback_shown': {}, 'show_result': False, 'current_question_options': {},
                'quiz_loaded': False
            })
            st.rerun() # Rerun to reflect change and potentially enable load button

    start_disabled = not st.session_state.selected_quiz_section_name
    if st.sidebar.button("Load / Restart Quiz", disabled=start_disabled, type="primary"):
        if st.session_state.selected_quiz_section_name:
            selected_df = quiz_sections.get(st.session_state.selected_quiz_section_name)
            if selected_df is not None and not selected_df.empty:
                # Ensure 'MedicationName' and 'IndicationFrench' columns exist for question generation
                if 'MedicationName' not in selected_df.columns or 'IndicationFrench' not in selected_df.columns:
                    st.sidebar.error("Selected data is missing 'MedicationName' or 'IndicationFrench' columns.")
                    st.session_state.quiz_loaded = False
                else:
                    st.session_state.current_quiz_df = selected_df.sample(frac=1).reset_index(drop=True)
                    st.session_state.update({
                        'question_index': 0, 'answers': {}, 'feedback_shown': {},
                        'show_result': False, 'current_question_options': {}, 'quiz_loaded': True
                    })
                    st.rerun()
            else:
                st.sidebar.error("No questions found for this section.")
                st.session_state.quiz_loaded = False
        else:
             st.sidebar.error("Please select a quiz section.")
             st.session_state.quiz_loaded = False
    st.sidebar.markdown("---")

    # --- Main Quiz Area ---
    st.title("💊 Pharm Mnemonics Quiz")
    st.markdown("---")

    if st.session_state.quiz_loaded and not st.session_state.show_result:
        df_current_quiz = st.session_state.current_quiz_df
        total_questions = len(df_current_quiz)
        current_q_index = st.session_state.question_index

        if 0 <= current_q_index < total_questions:
            question_data = df_current_quiz.iloc[current_q_index]
            correct_med_name = question_data['MedicationName']
            correct_indication = question_data['IndicationFrench']
            correct_mnemonic = question_data.get('MnemonicText', "No mnemonic available for this medication.")

            # --- Image Display ---
            # Assuming 'Category' column exists in your Mnemonics.xlsx or was added
            # If not, you might need a default category or adjust get_image_url
            img_category_for_lookup = question_data.get('Category', "Generic Drugs") # Fallback category
            image_url = get_image_url(df_images, img_category_for_lookup, correct_med_name)

            left_col, right_col = st.columns([2, 1])

            with right_col:
                st.subheader(f"Medication: {correct_med_name}") # Show med name with image
                display_image(image_url, st) # Pass st or right_col

            with left_col:
                st.subheader(f"Question {current_q_index + 1} of {total_questions}")
                question_text = f"Quel médicament est utile en cas de **{correct_indication}** ?"
                st.markdown(f"#### {question_text}")

                # --- Generate Options ---
                if current_q_index not in st.session_state.current_question_options:
                    options = [correct_med_name]
                    # Get 4 wrong options with different indications
                    # Ensure df_mnemonics_all is available and has the necessary columns
                    if df_mnemonics_all is not None and not df_mnemonics_all.empty and \
                       'MedicationName' in df_mnemonics_all.columns and 'IndicationFrench' in df_mnemonics_all.columns:
                        
                        wrong_options_pool = df_mnemonics_all[
                            (df_mnemonics_all['MedicationName'] != correct_med_name) &
                            (df_mnemonics_all['IndicationFrench'] != correct_indication)
                        ]['MedicationName'].unique()
                        
                        if len(wrong_options_pool) >= 4:
                            options.extend(random.sample(list(wrong_options_pool), 4))
                        elif len(wrong_options_pool) > 0 : # If less than 4, take all available
                            options.extend(list(wrong_options_pool))
                            # If still not 5 options, you might need a more complex fallback
                            # For simplicity, we'll proceed with fewer if necessary, or you can duplicate
                            while len(options) < 5 and len(wrong_options_pool) > 0:
                                options.append(random.choice(list(wrong_options_pool))) # allow duplicates if desperate
                        else: # Very few unique drugs, may need to allow same indication from other drugs
                             st.warning("Not enough unique wrong options with different indications, options may be less diverse.")
                             fallback_pool = df_mnemonics_all[df_mnemonics_all['MedicationName'] != correct_med_name]['MedicationName'].unique()
                             if len(fallback_pool) > 0:
                                 needed = 5 - len(options)
                                 options.extend(random.sample(list(fallback_pool), min(needed, len(fallback_pool))))


                    else: # Fallback if full mnemonics data isn't loaded for options generation
                        st.error("Full mnemonics data not available for generating diverse options.")
                        # Add some placeholder wrong options if the main pool fails
                        placeholders = [f"Option Fausse {i}" for i in range(1, 5)]
                        options.extend(placeholders[:(5-len(options))])


                    while len(options) < 5 : # Ensure 5 options, even if with placeholders
                        options.append(f"Placeholder Option {len(options)}")


                    random.shuffle(options)
                    st.session_state.current_question_options[current_q_index] = options
                else:
                    options = st.session_state.current_question_options[current_q_index]

                # --- Display Radio Buttons ---
                previous_answer_val = st.session_state.answers.get(current_q_index)
                default_option_idx = None
                if previous_answer_val is not None and previous_answer_val in options:
                    default_option_idx = options.index(previous_answer_val)

                user_choice = st.radio(
                    "Sélectionnez votre réponse:",
                    options=options,
                    index=default_option_idx,
                    key=f"q_{current_q_index}_options",
                    disabled=st.session_state.feedback_shown.get(current_q_index, False) # Disable if feedback shown
                )

                submit_pressed = st.button("Soumettre la réponse", key=f"submit_{current_q_index}",
                                           disabled=st.session_state.feedback_shown.get(current_q_index, False))

                if submit_pressed:
                    st.session_state.answers[current_q_index] = user_choice
                    st.session_state.feedback_shown[current_q_index] = True # Mark as feedback shown
                    st.rerun()

                # --- Display Feedback and Mnemonic ---
                if st.session_state.feedback_shown.get(current_q_index, False):
                    stored_answer = st.session_state.answers[current_q_index]
                    if stored_answer == correct_med_name:
                        st.success(f"Correct! ✅")
                    else:
                        st.error(f"Incorrect! ❌ La bonne réponse est : **{correct_med_name}**")
                    st.info(f"💡 Mnémonique: {correct_mnemonic}")


            # --- Navigation Buttons (BELOW columns) ---
            st.markdown("---")
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("⬅️ Précédent", disabled=current_q_index <= 0, use_container_width=True):
                    st.session_state.question_index -= 1
                    # st.session_state.feedback_shown[st.session_state.question_index + 1] = False # Optional: allow re-answering previous
                    st.rerun()
            with nav_col2:
                if st.button("Suivant ➡️", disabled=current_q_index >= total_questions - 1 or \
                                           not st.session_state.feedback_shown.get(current_q_index, False), # Only allow next if current answered
                            use_container_width=True):
                    st.session_state.question_index += 1
                    st.rerun()
            
            st.markdown("---")
            st.write("**Aller à la question :**")
            num_nav_cols = min(total_questions, 10) # Display up to 10 nav buttons per row
            nav_buttons_cols = st.columns(num_nav_cols)
            for i in range(total_questions):
                col_to_use = nav_buttons_cols[i % num_nav_cols]
                q_label_nav = str(i + 1)
                q_state_icon_nav = ""
                if i in st.session_state.answers: # Check if an answer was submitted
                    is_correct_nav = st.session_state.answers[i] == df_current_quiz.iloc[i]['MedicationName']
                    q_state_icon_nav = " ✅" if is_correct_nav else " ❌"
                
                btn_type_nav = "primary" if i == current_q_index else "secondary"
                
                # Disable navigation to questions not yet "unlocked" by answering previous ones, if desired
                # nav_disabled = i > current_q_index and not st.session_state.feedback_shown.get(current_q_index, False)
                
                if col_to_use.button(f"{q_label_nav}{q_state_icon_nav}", key=f"nav_{i}", type=btn_type_nav, use_container_width=True):
                    st.session_state.question_index = i
                    st.rerun()


            st.markdown("---")
            if st.button("🏁 Terminer le Quiz et Voir les Résultats", use_container_width=True,
                        disabled=not all(st.session_state.feedback_shown.get(i, False) for i in range(total_questions))): # Ensure all questions attempted
                st.session_state.show_result = True
                st.rerun()
        else: # Invalid question index
            st.warning("Index de question invalide. Redémarrage de la sélection du quiz.")
            st.session_state.question_index = 0
            st.session_state.quiz_loaded = False
            st.rerun()

    # --- Results Page ---
    elif st.session_state.show_result:
        st.subheader("📊 Résultats du Quiz")
        df_quiz_completed = st.session_state.current_quiz_df # Use the df from the completed quiz
        total_q_completed = len(df_quiz_completed)
        answers_completed = st.session_state.answers
        correct_count = 0
        
        # Calculate score
        for i in range(total_q_completed):
            if i in answers_completed and answers_completed[i] == df_quiz_completed.iloc[i]['MedicationName']:
                correct_count += 1
        
        incorrect_count = len(answers_completed) - correct_count # Count answered incorrect questions
        unanswered_count = total_q_completed - len(answers_completed) # Should be 0 if Finish button logic is strict

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("✅ Correctes", correct_count)
        res_col2.metric("❌ Incorrectes", incorrect_count)
        res_col3.metric("❓ Non répondues", unanswered_count)
        score_percent_val = (correct_count / total_q_completed * 100) if total_q_completed > 0 else 0
        res_col4.metric("🏆 Score", f"{correct_count}/{total_q_completed}", f"{score_percent_val:.1f}%")

        results_btn_col1, results_btn_col2 = st.columns(2)
        with results_btn_col1:
            if st.button("🚀 Recommencer ce Quiz (Mêmes Paramètres)", use_container_width=True):
                # Reshuffle the same section's DataFrame
                st.session_state.current_quiz_df = quiz_sections.get(st.session_state.selected_quiz_section_name).sample(frac=1).reset_index(drop=True)
                st.session_state.update({
                    'question_index': 0, 'answers': {}, 'feedback_shown': {},
                    'show_result': False, 'current_question_options': {}, 'quiz_loaded': True
                })
                st.rerun()
        with results_btn_col2:
            if st.button("⚙️ Changer de Section et Recommencer", use_container_width=True):
                st.session_state.update({
                    # 'selected_quiz_section_name': None, # Optional: clear section selection
                    'question_index': 0, 'answers': {}, 'feedback_shown': {},
                    'show_result': False, 'current_quiz_df': pd.DataFrame(),
                    'current_question_options': {}, 'quiz_loaded': False
                })
                st.rerun()

        # --- Review Answers Expander ---
        with st.expander("🧐 Revoir Vos Réponses", expanded=False):
            if total_q_completed == 0: st.write("Aucune question n'a été chargée pour la révision.")
            else:
                for i in range(total_q_completed):
                    q_data_review = df_quiz_completed.iloc[i]
                    user_ans_review = answers_completed.get(i, "Non Répondu")
                    correct_ans_review = q_data_review['MedicationName']
                    is_correct_review = user_ans_review == correct_ans_review
                    mnemonic_review = q_data_review.get('MnemonicText', "Pas de mnémonique disponible.")
                    indication_review = q_data_review['IndicationFrench']

                    status_icon_review = ""
                    if i in answers_completed: status_icon_review = "✅" if is_correct_review else "❌"
                    else: status_icon_review = "❓"

                    st.markdown(f"**Question {i+1}:** Quel médicament est utile en cas de **{indication_review}** ?")
                    
                    # Image in review
                    review_img_cat = q_data_review.get('Category', "Generic Drugs")
                    review_med_name = q_data_review['MedicationName']
                    review_image_url = get_image_url(df_images, review_img_cat, review_med_name)
                    
                    # Layout for review item (image on right, text on left)
                    rev_left_col, rev_right_col = st.columns([3,1])
                    with rev_right_col:
                         display_image(review_image_url, st)
                    with rev_left_col:
                        st.write(f"Votre réponse : **{user_ans_review}** {status_icon_review}")
                        if not is_correct_review and i in answers_completed:
                            st.write(f"Bonne réponse : **{correct_ans_review}**")
                        elif user_ans_review == "Non Répondu":
                            st.write(f"Bonne réponse : **{correct_ans_review}**")
                        st.info(f"💡 Mnémonique: {mnemonic_review}")
                    st.divider()

    # --- Initial State Message ---
    elif not st.session_state.quiz_loaded:
        st.info("👋 Bienvenue ! Veuillez sélectionner une section de quiz dans la barre latérale, puis cliquez sur 'Charger / Recommencer le Quiz'.")