import streamlit as st
import pandas as pd
import random
import unicodedata
import re # Keep for potential future use, though not directly used in current normalization
from pathlib import Path

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Pharm Mnemonics Quiz")

# --- Project Setup & File Paths ---
try:
    APP_DIR = Path(__file__).parent
except NameError:
    APP_DIR = Path.cwd()

PROJECT_ROOT = APP_DIR
# IMPORTANT: If Mnemonics.xlsx is NOT in your project's QUIZ folder,
# update this path to its absolute location or ensure it's copied.
# For portability, having it in the project is best.
MNEMONICS_XLSX_PATH = PROJECT_ROOT / "QUIZ" / "Mnemonics.xlsx"
IMAGE_CSV_PATH = PROJECT_ROOT / "QUIZ" / "github_image_urls_CATEGORIZED.csv"

# --- Load Mnemonics Data ---
@st.cache_data(show_spinner="Loading mnemonics data...")
def load_mnemonics_data(path):
    try:
        if not path.exists():
            st.error(f"Mnemonics data XLSX not found: {path}")
            st.error("Please ensure 'Mnemonics.xlsx' is in the 'QUIZ' subfolder of your project, or update MNEMONICS_XLSX_PATH in the script.")
            return None
        df = pd.read_excel(path, keep_default_na=False) # openpyxl engine needed (add to requirements.txt)
        required_mnemonics = ['Commercial Name', 'Indication (French Keyword)', 'Mnemonic']
        if not all(col in df.columns for col in required_mnemonics):
            st.error(f"Mnemonics XLSX missing required columns. Need: {required_mnemonics}. Found: {list(df.columns)}")
            return None

        df.rename(columns={'Commercial Name': 'MedicationName',
                           'Indication (French Keyword)': 'IndicationFrench',
                           'Mnemonic': 'MnemonicText'},
                  inplace=True)
        # Ensure 'Category' exists, for image lookup compatibility
        if 'Category' not in df.columns:
            df['Category'] = "Generic Drugs" # Default category if not present
        return df
    except Exception as e:
        st.error(f"Failed to load/parse mnemonics XLSX: {e}")
        st.error("Make sure 'openpyxl' is listed in your requirements.txt file for Streamlit Cloud.")
        return None

# --- Load Image URL Data ---
@st.cache_data(show_spinner="Loading image data...")
def load_image_data(path):
    try:
        if not path.exists():
            st.warning(f"Image data CSV not found: {path}. Images might not load.")
            return pd.DataFrame(columns=['category', 'filename', 'raw_url', '_norm_cat', '_norm_filename'])
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

        df['_norm_cat'] = df['category'].astype(str).apply(normalize_text)
        df['_norm_filename'] = df['filename'].astype(str).str.extract(r'([^/\\.]+)(?:\.[^.]*$|$)', expand=False).fillna('').apply(normalize_text)
        return df
    except Exception as e:
        st.error(f"Error loading image CSV: {e}")
        return pd.DataFrame(columns=['category', 'filename', 'raw_url', '_norm_cat', '_norm_filename'])

# --- Normalize function for matching ---
def normalize_text(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text.lower().strip()

# --- Get Image URL ---
def get_image_url(df_images, category, medication_name):
    if df_images.empty or not medication_name or not category: return None

    norm_med = normalize_text(medication_name)
    norm_cat = normalize_text(category)

    img_row = df_images[
        (df_images['_norm_cat'] == norm_cat) &
        (df_images['_norm_filename'] == norm_med)
    ]

    if not img_row.empty:
        return img_row.iloc[0]['raw_url']
    
    # Fallback: try matching just by medication name
    img_row_med_only = df_images[df_images['_norm_filename'] == norm_med]
    if not img_row_med_only.empty:
        # st.sidebar.warning(f"Note: Image for '{medication_name}' found by name only (category '{category}' did not yield specific match).")
        return img_row_med_only.iloc[0]['raw_url']
    return None

# --- Display Image ---
def display_image(url, container):
    if url:
        container.markdown(f'<a href="{url}" target="_blank"><img src="{url}" alt="Medication Image" style="max-width:100%; max-height: 350px; object-fit: contain; margin-top: 10px; border-radius: 5px;"/></a>', unsafe_allow_html=True)

# --- Sectioning Logic ---
NUM_QUIZ_SECTIONS = 15
def get_quiz_sections(df_mnemonics_full):
    if df_mnemonics_full is None or df_mnemonics_full.empty:
        return {}
    total_meds = len(df_mnemonics_full)
    meds_per_section = (total_meds + NUM_QUIZ_SECTIONS - 1) // NUM_QUIZ_SECTIONS
    sections = {}
    for i in range(NUM_QUIZ_SECTIONS):
        start_idx = i * meds_per_section
        end_idx = min((i + 1) * meds_per_section, total_meds)
        if start_idx < end_idx:
            section_name = f"Section {i+1} (Médicaments {start_idx+1}-{end_idx})"
            sections[section_name] = df_mnemonics_full.iloc[start_idx:end_idx].copy()
    return sections

# --- Main Execution ---
if __name__ == "__main__":
    df_mnemonics_all = load_mnemonics_data(MNEMONICS_XLSX_PATH)
    df_images = load_image_data(IMAGE_CSV_PATH)

    if df_mnemonics_all is None:
        # Error messages are handled within load_mnemonics_data
        st.stop()

    quiz_sections_dict = get_quiz_sections(df_mnemonics_all)

    default_session_values = {
        'selected_quiz_section_name': None, 'question_index': 0, 'answers': {},
        'feedback_shown': {}, 'show_result': False, 'current_quiz_df': pd.DataFrame(),
        'current_question_options': {}, 'quiz_loaded': False
    }
    for key, default_value in default_session_values.items():
        st.session_state.setdefault(key, default_value)

    st.sidebar.title("Paramètres du Quiz")
    if not quiz_sections_dict:
        st.sidebar.error("Aucune section de quiz disponible.")
    else:
        section_names_list = list(quiz_sections_dict.keys())
        current_sel_section = st.session_state.selected_quiz_section_name
        if current_sel_section not in section_names_list and section_names_list:
            current_sel_section = section_names_list[0]
        
        selected_section_name_from_sb = st.sidebar.selectbox(
            "Sélectionnez une Section de Quiz", options=section_names_list,
            index=section_names_list.index(current_sel_section) if current_sel_section in section_names_list else 0,
            key='sb_section_selector'
        )

        if selected_section_name_from_sb != st.session_state.selected_quiz_section_name:
            st.session_state.selected_quiz_section_name = selected_section_name_from_sb
            st.session_state.update({
                'current_quiz_df': pd.DataFrame(), 'question_index': 0, 'answers': {},
                'feedback_shown': {}, 'show_result': False, 'current_question_options': {},
                'quiz_loaded': False
            })
            st.rerun()

    start_btn_disabled = not st.session_state.selected_quiz_section_name
    if st.sidebar.button("Charger / Recommencer le Quiz", disabled=start_btn_disabled, type="primary", use_container_width=True):
        if st.session_state.selected_quiz_section_name:
            selected_section_df = quiz_sections_dict.get(st.session_state.selected_quiz_section_name)
            if selected_section_df is not None and not selected_section_df.empty:
                if 'MedicationName' not in selected_section_df.columns or 'IndicationFrench' not in selected_section_df.columns:
                    st.sidebar.error("Les données sélectionnées manquent les colonnes 'MedicationName' ou 'IndicationFrench'.")
                    st.session_state.quiz_loaded = False
                else:
                    st.session_state.current_quiz_df = selected_section_df.sample(frac=1).reset_index(drop=True)
                    st.session_state.update({
                        'question_index': 0, 'answers': {}, 'feedback_shown': {},
                        'show_result': False, 'current_question_options': {}, 'quiz_loaded': True
                    })
                    st.rerun()
            else:
                st.sidebar.error("Aucune question trouvée pour cette section.")
                st.session_state.quiz_loaded = False
        else:
             st.sidebar.error("Veuillez sélectionner une section de quiz.")
             st.session_state.quiz_loaded = False
    st.sidebar.markdown("---")

    st.title("💊 Quiz Mnémoniques Pharma")
    st.markdown("---")

    if st.session_state.quiz_loaded and not st.session_state.show_result:
        df_current_quiz_active = st.session_state.current_quiz_df
        total_questions_active = len(df_current_quiz_active)
        current_q_idx_active = st.session_state.question_index

        if 0 <= current_q_idx_active < total_questions_active:
            question_data_active = df_current_quiz_active.iloc[current_q_idx_active]
            correct_med_name_active = question_data_active['MedicationName']
            correct_indication_active = question_data_active['IndicationFrench']
            correct_mnemonic_active = question_data_active.get('MnemonicText', "Pas de mnémonique disponible.")
            
            img_category_active = question_data_active.get('Category', "Generic Drugs")
            image_url_active = get_image_url(df_images, img_category_active, correct_med_name_active)

            q_left_col, q_right_col = st.columns([2, 1])

            with q_right_col:
                if st.session_state.feedback_shown.get(current_q_idx_active, False):
                    st.subheader(f"Médicament: {correct_med_name_active}")
                    display_image(image_url_active, st)
                else:
                    st.write("_L'image du médicament apparaîtra après la soumission._")

            with q_left_col:
                st.subheader(f"Question {current_q_idx_active + 1} sur {total_questions_active}")
                question_text_active = f"Quel médicament est utile en cas de **{correct_indication_active}** ?"
                st.markdown(f"#### {question_text_active}")

                if current_q_idx_active not in st.session_state.current_question_options:
                    options_list = [correct_med_name_active]
                    if df_mnemonics_all is not None and not df_mnemonics_all.empty:
                        wrong_options_pool_df = df_mnemonics_all[
                            (df_mnemonics_all['MedicationName'] != correct_med_name_active) &
                            (df_mnemonics_all['IndicationFrench'] != correct_indication_active)
                        ]
                        if not wrong_options_pool_df.empty:
                             wrong_med_names = wrong_options_pool_df['MedicationName'].unique()
                             num_to_sample = min(4, len(wrong_med_names))
                             if num_to_sample > 0 :
                                options_list.extend(random.sample(list(wrong_med_names), num_to_sample))
                        
                        if len(options_list) < 5: # Fallback if not enough diverse indications
                            fallback_pool_df = df_mnemonics_all[df_mnemonics_all['MedicationName'] != correct_med_name_active]
                            if not fallback_pool_df.empty:
                                fallback_med_names = fallback_pool_df['MedicationName'].unique()
                                needed_fallback = 5 - len(options_list)
                                num_fallback_sample = min(needed_fallback, len(fallback_med_names))
                                if num_fallback_sample > 0:
                                    options_list.extend(random.sample(list(fallback_med_names), num_fallback_sample))
                    
                    while len(options_list) < 5: # Ensure 5 options with placeholders if necessary
                        options_list.append(f"Option Placeholder {len(options_list)}")
                    
                    random.shuffle(options_list)
                    st.session_state.current_question_options[current_q_idx_active] = options_list
                else:
                    options_list = st.session_state.current_question_options[current_q_idx_active]

                previous_ans_val = st.session_state.answers.get(current_q_idx_active)
                default_opt_idx = None
                if previous_ans_val is not None and previous_ans_val in options_list:
                    try: default_opt_idx = options_list.index(previous_ans_val)
                    except ValueError: pass
                
                user_choice_active = st.radio(
                    "Sélectionnez votre réponse:", options=options_list, index=default_opt_idx,
                    key=f"q_radio_{current_q_idx_active}",
                    disabled=st.session_state.feedback_shown.get(current_q_idx_active, False)
                )

                submit_btn_pressed = st.button("Soumettre la réponse", key=f"submit_btn_{current_q_idx_active}",
                                               disabled=st.session_state.feedback_shown.get(current_q_idx_active, False))

                if submit_btn_pressed:
                    st.session_state.answers[current_q_idx_active] = user_choice_active
                    st.session_state.feedback_shown[current_q_idx_active] = True
                    st.rerun()

                if st.session_state.feedback_shown.get(current_q_idx_active, False):
                    stored_ans_active = st.session_state.answers.get(current_q_idx_active)
                    if stored_ans_active == correct_med_name_active:
                        st.success("Correct! ✅")
                    else:
                        st.error(f"Incorrect! ❌ La bonne réponse est : **{correct_med_name_active}**")
                    st.info(f"💡 Mnémonique: {correct_mnemonic_active}")
            
            st.markdown("---")
            nav_prev_col, nav_next_col = st.columns(2)
            with nav_prev_col:
                if st.button("⬅️ Précédent", disabled=current_q_idx_active <= 0, use_container_width=True):
                    st.session_state.question_index -= 1
                    st.rerun()
            with nav_next_col:
                next_btn_disabled = current_q_idx_active >= total_questions_active - 1 or \
                                    not st.session_state.feedback_shown.get(current_q_idx_active, False)
                if st.button("Suivant ➡️", disabled=next_btn_disabled, use_container_width=True):
                    st.session_state.question_index += 1
                    st.rerun()
            
            st.markdown("---")
            st.write("**Aller à la question :**")
            num_quick_nav_cols = min(total_questions_active, 10)
            quick_nav_cols = st.columns(num_quick_nav_cols)
            for i_nav in range(total_questions_active):
                col_for_nav_btn = quick_nav_cols[i_nav % num_quick_nav_cols]
                nav_label = str(i_nav + 1)
                nav_icon = ""
                if i_nav in st.session_state.answers:
                    is_correct_nav_q = st.session_state.answers[i_nav] == df_current_quiz_active.iloc[i_nav]['MedicationName']
                    nav_icon = " ✅" if is_correct_nav_q else " ❌"
                
                nav_btn_type = "primary" if i_nav == current_q_idx_active else "secondary"
                if col_for_nav_btn.button(f"{nav_label}{nav_icon}", key=f"quick_nav_{i_nav}", type=nav_btn_type, use_container_width=True):
                    st.session_state.question_index = i_nav
                    st.rerun()

            st.markdown("---")
            all_q_attempted = all(st.session_state.feedback_shown.get(i, False) for i in range(total_questions_active))
            if st.button("🏁 Terminer le Quiz et Voir les Résultats", use_container_width=True, disabled=not all_q_attempted):
                st.session_state.show_result = True
                st.rerun()
        else:
            st.warning("Index de question invalide. Redémarrage de la sélection du quiz.")
            st.session_state.question_index = 0
            st.session_state.quiz_loaded = False
            st.rerun()

    elif st.session_state.show_result:
        st.subheader("📊 Résultats du Quiz")
        df_quiz_review = st.session_state.current_quiz_df
        total_q_review = len(df_quiz_review)
        answers_review = st.session_state.answers
        correct_count_review = 0
        
        for i_rev in range(total_q_review):
            if i_rev in answers_review and answers_review[i_rev] == df_quiz_review.iloc[i_rev]['MedicationName']:
                correct_count_review += 1
        
        incorrect_count_review = len([ans for i, ans in answers_review.items() if i < total_q_review and ans != df_quiz_review.iloc[i]['MedicationName']])
        unanswered_review = total_q_review - len([i for i in answers_review.keys() if i < total_q_review])


        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("✅ Correctes", correct_count_review)
        stat_col2.metric("❌ Incorrectes", incorrect_count_review)
        stat_col3.metric("❓ Non répondues", unanswered_review)
        score_val = (correct_count_review / total_q_review * 100) if total_q_review > 0 else 0
        stat_col4.metric("🏆 Score", f"{correct_count_review}/{total_q_review}", f"{score_val:.1f}%")

        res_btn_col_1, res_btn_col_2 = st.columns(2)
        with res_btn_col_1:
            if st.button("🚀 Recommencer ce Quiz", use_container_width=True):
                current_section_data = quiz_sections_dict.get(st.session_state.selected_quiz_section_name)
                if current_section_data is not None:
                    st.session_state.current_quiz_df = current_section_data.sample(frac=1).reset_index(drop=True)
                st.session_state.update({
                    'question_index': 0, 'answers': {}, 'feedback_shown': {},
                    'show_result': False, 'current_question_options': {}, 'quiz_loaded': True
                })
                st.rerun()
        with res_btn_col_2:
            if st.button("⚙️ Changer de Section", use_container_width=True):
                st.session_state.update({
                    'question_index': 0, 'answers': {}, 'feedback_shown': {},
                    'show_result': False, 'current_quiz_df': pd.DataFrame(),
                    'current_question_options': {}, 'quiz_loaded': False
                })
                st.rerun()

        with st.expander("🧐 Revoir Vos Réponses", expanded=False):
            if total_q_review == 0: st.write("Aucune question n'a été chargée pour la révision.")
            else:
                for i_detail in range(total_q_review):
                    q_data_detail = df_quiz_review.iloc[i_detail]
                    user_ans_detail = answers_review.get(i_detail, "Non Répondu")
                    correct_ans_detail = q_data_detail['MedicationName']
                    is_correct_detail = user_ans_detail == correct_ans_detail
                    mnemonic_detail = q_data_detail.get('MnemonicText', "Pas de mnémonique disponible.")
                    indication_detail = q_data_detail['IndicationFrench']
                    status_icon_detail = "❓"
                    if i_detail in answers_review: status_icon_detail = "✅" if is_correct_detail else "❌"

                    st.markdown(f"**Question {i_detail+1}:** Quel médicament est utile en cas de **{indication_detail}** ?")
                    
                    review_img_cat_detail = q_data_detail.get('Category', "Generic Drugs")
                    review_med_name_detail = q_data_detail['MedicationName']
                    review_img_url_detail = get_image_url(df_images, review_img_cat_detail, review_med_name_detail)
                    
                    rev_detail_left, rev_detail_right = st.columns([3,1])
                    with rev_detail_right:
                         display_image(review_img_url_detail, st) # Or pass rev_detail_right
                    with rev_detail_left:
                        st.write(f"Votre réponse : **{user_ans_detail}** {status_icon_detail}")
                        if not is_correct_detail and i_detail in answers_review:
                            st.write(f"Bonne réponse : **{correct_ans_detail}**")
                        elif user_ans_detail == "Non Répondu": # Also show correct answer if unanswered
                            st.write(f"Bonne réponse : **{correct_ans_detail}**")
                        st.info(f"💡 Mnémonique: {mnemonic_detail}")
                    st.divider()

    elif not st.session_state.quiz_loaded:
        st.info("👋 Bienvenue ! Veuillez sélectionner une section de quiz dans la barre latérale, puis cliquez sur 'Charger / Recommencer le Quiz'.")
