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
    APP_DIR = Path.cwd()

PROJECT_ROOT = APP_DIR
# --- Point to the PRESHUFFLED Mnemonics file ---
MNEMONICS_XLSX_PATH = PROJECT_ROOT / "QUIZ" / "Mnemonics_PRESHUFFLED.xlsx"
IMAGE_CSV_PATH = PROJECT_ROOT / "QUIZ" / "github_image_urls_CATEGORIZED.csv"

# --- Load Mnemonics Data (No master shuffle here anymore) ---
@st.cache_data(show_spinner="Loading mnemonics data...")
def load_mnemonics_data(path):
    try:
        if not path.exists():
            st.error(f"Mnemonics data XLSX not found: {path}")
            st.error("Please ensure 'Mnemonics_PRESHUFFLED.xlsx' is in the 'QUIZ' subfolder of your project.")
            return None
        df = pd.read_excel(path, keep_default_na=False) # openpyxl engine needed
        required_mnemonics = ['Commercial Name', 'Indication (French Keyword)', 'Mnemonic']
        if not all(col in df.columns for col in required_mnemonics):
            st.error(f"Mnemonics XLSX missing required columns. Need: {required_mnemonics}. Found: {list(df.columns)}")
            return None

        df.rename(columns={'Commercial Name': 'MedicationName',
                           'Indication (French Keyword)': 'IndicationFrench',
                           'Mnemonic': 'MnemonicText'},
                  inplace=True)
        if 'Category' not in df.columns: # Ensure 'Category' exists for image lookup compatibility
            df['Category'] = "Generic Drugs" # Default category if not present
        
        # The DataFrame read from Mnemonics_PRESHUFFLED.xlsx is already in the desired "master" order.
        # No further df.sample(frac=1) is needed here for section consistency.
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
            st.warning(f"UTF-8 decoding failed for {path}, trying UTF-8-SIG...")
            df = pd.read_csv(path, encoding='utf-8-sig')
        
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
def get_image_url(df_images_data, category_name, medication_name_str):
    if df_images_data.empty or not medication_name_str or not category_name: return None
    norm_med_name = normalize_text(medication_name_str)
    norm_category_name = normalize_text(category_name)
    img_row_match = df_images_data[
        (df_images_data['_norm_cat'] == norm_category_name) &
        (df_images_data['_norm_filename'] == norm_med_name)
    ]
    if not img_row_match.empty:
        return img_row_match.iloc[0]['raw_url']
    img_row_med_only_match = df_images_data[df_images_data['_norm_filename'] == norm_med_name]
    if not img_row_med_only_match.empty:
        return img_row_med_only_match.iloc[0]['raw_url']
    return None

# --- Display Image ---
def display_image(image_display_url, container_obj):
    if image_display_url:
        container_obj.markdown(f'<a href="{image_display_url}" target="_blank"><img src="{image_display_url}" alt="Medication Image" style="max-width:100%; max-height: 350px; object-fit: contain; margin-top: 10px; border-radius: 5px;"/></a>', unsafe_allow_html=True)

# --- Sectioning Logic (operates on the pre-shuffled master list) ---
NUM_QUIZ_SECTIONS = 15
def get_quiz_sections(df_ordered_master_list): # Parameter name reflects it's the master order
    if df_ordered_master_list is None or df_ordered_master_list.empty:
        return {}
    total_meds_count = len(df_ordered_master_list)
    meds_per_section_count = (total_meds_count + NUM_QUIZ_SECTIONS - 1) // NUM_QUIZ_SECTIONS
    quiz_sections_map = {}
    for i_section in range(NUM_QUIZ_SECTIONS):
        start_index = i_section * meds_per_section_count
        end_index = min((i_section + 1) * meds_per_section_count, total_meds_count)
        if start_index < end_index:
            # Section title reflects that the content is now from a fixed, pre-shuffled master
            section_title = f"Section {i_section+1} (Ensemble Fixe {start_index+1}-{end_index})"
            quiz_sections_map[section_title] = df_ordered_master_list.iloc[start_index:end_index].copy()
    return quiz_sections_map

# --- Main Execution ---
if __name__ == "__main__":
    # df_mnemonics_all is now loaded from the pre-shuffled file
    df_mnemonics_all = load_mnemonics_data(MNEMONICS_XLSX_PATH)
    df_images = load_image_data(IMAGE_CSV_PATH)

    if df_mnemonics_all is None:
        st.stop()

    quiz_sections_dict_main = get_quiz_sections(df_mnemonics_all)

    default_session_vals_main = {
        'selected_quiz_section_name': None, 'question_index': 0, 'answers': {},
        'feedback_shown': {}, 'show_result': False, 'current_quiz_df': pd.DataFrame(),
        'current_question_options': {}, 'quiz_loaded': False
    }
    for key_state, default_val_state in default_session_vals_main.items():
        st.session_state.setdefault(key_state, default_val_state)

    st.sidebar.title("Paramètres du Quiz")
    if not quiz_sections_dict_main:
        st.sidebar.error("Aucune section de quiz disponible.")
    else:
        section_names_list_main = list(quiz_sections_dict_main.keys())
        current_sel_section_main = st.session_state.selected_quiz_section_name
        if current_sel_section_main not in section_names_list_main and section_names_list_main:
            current_sel_section_main = section_names_list_main[0]
        
        selected_section_sb_main = st.sidebar.selectbox(
            "Sélectionnez une Section de Quiz", options=section_names_list_main,
            index=section_names_list_main.index(current_sel_section_main) if current_sel_section_main in section_names_list_main else 0,
            key='sb_section_selector_main'
        )

        if selected_section_sb_main != st.session_state.selected_quiz_section_name:
            st.session_state.selected_quiz_section_name = selected_section_sb_main
            st.session_state.update({
                'current_quiz_df': pd.DataFrame(), 'question_index': 0, 'answers': {},
                'feedback_shown': {}, 'show_result': False, 'current_question_options': {},
                'quiz_loaded': False
            })
            st.rerun()

    start_btn_disabled_main = not st.session_state.selected_quiz_section_name
    if st.sidebar.button("Charger / Recommencer le Quiz", disabled=start_btn_disabled_main, type="primary", use_container_width=True):
        if st.session_state.selected_quiz_section_name:
            # Get the section DataFrame (which is from the pre-shuffled master list)
            section_df_for_quiz = quiz_sections_dict_main.get(st.session_state.selected_quiz_section_name)
            
            if section_df_for_quiz is not None and not section_df_for_quiz.empty:
                if 'MedicationName' not in section_df_for_quiz.columns or 'IndicationFrench' not in section_df_for_quiz.columns:
                    st.sidebar.error("Les données sélectionnées manquent les colonnes 'MedicationName' ou 'IndicationFrench'.")
                    st.session_state.quiz_loaded = False
                else:
                    # Now, shuffle the questions *within this section* for the current quiz attempt
                    st.session_state.current_quiz_df = section_df_for_quiz.copy().sample(frac=1).reset_index(drop=True)
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
        df_current_quiz_active_main = st.session_state.current_quiz_df
        total_questions_active_main = len(df_current_quiz_active_main)
        current_q_idx_active_main = st.session_state.question_index

        if 0 <= current_q_idx_active_main < total_questions_active_main:
            question_data_active_main = df_current_quiz_active_main.iloc[current_q_idx_active_main]
            correct_med_name_active_main = question_data_active_main['MedicationName']
            correct_indication_active_main = question_data_active_main['IndicationFrench']
            raw_mnemonic_active_main = question_data_active_main.get('MnemonicText', "Pas de mnémonique disponible.")
            processed_mnemonic_active_main = re.sub(r'##(.*?)##', r'**\1**', raw_mnemonic_active_main)
            
            img_category_active_main = question_data_active_main.get('Category', "Generic Drugs")
            image_url_active_main = get_image_url(df_images, img_category_active_main, correct_med_name_active_main)

            q_left_col_main, q_right_col_main = st.columns([2, 1])

            with q_right_col_main:
                if st.session_state.feedback_shown.get(current_q_idx_active_main, False):
                    st.subheader(f"Médicament: {correct_med_name_active_main}")
                    display_image(image_url_active_main, st)
                else:
                    st.write("_L'image du médicament apparaîtra après la soumission._")

            with q_left_col_main:
                st.subheader(f"Question {current_q_idx_active_main + 1} sur {total_questions_active_main}")
                question_text_active_main = f"Quel médicament est utile en cas de **{correct_indication_active_main}** ?"
                st.markdown(f"#### {question_text_active_main}")

                if current_q_idx_active_main not in st.session_state.current_question_options:
                    options_list_main = [correct_med_name_active_main]
                    if df_mnemonics_all is not None and not df_mnemonics_all.empty: # Use df_mnemonics_all for broad option pool
                        wrong_options_pool_df_main = df_mnemonics_all[
                            (df_mnemonics_all['MedicationName'] != correct_med_name_active_main) &
                            (df_mnemonics_all['IndicationFrench'] != correct_indication_active_main)
                        ]
                        if not wrong_options_pool_df_main.empty:
                             wrong_med_names_main = wrong_options_pool_df_main['MedicationName'].unique()
                             num_to_sample_main = min(4, len(wrong_med_names_main))
                             if num_to_sample_main > 0 :
                                options_list_main.extend(random.sample(list(wrong_med_names_main), num_to_sample_main))
                        
                        if len(options_list_main) < 5:
                            fallback_pool_df_main = df_mnemonics_all[df_mnemonics_all['MedicationName'] != correct_med_name_active_main]
                            if not fallback_pool_df_main.empty:
                                fallback_med_names_main = fallback_pool_df_main['MedicationName'].unique()
                                needed_fallback_main = 5 - len(options_list_main)
                                num_fallback_sample_main = min(needed_fallback_main, len(fallback_med_names_main))
                                if num_fallback_sample_main > 0:
                                    options_list_main.extend(random.sample(list(fallback_med_names_main), num_fallback_sample_main))
                    
                    while len(options_list_main) < 5:
                        options_list_main.append(f"Option Placeholder {len(options_list_main)}")
                    
                    random.shuffle(options_list_main)
                    st.session_state.current_question_options[current_q_idx_active_main] = options_list_main
                else:
                    options_list_main = st.session_state.current_question_options[current_q_idx_active_main]

                previous_ans_val_main = st.session_state.answers.get(current_q_idx_active_main)
                default_opt_idx_main = None
                if previous_ans_val_main is not None and previous_ans_val_main in options_list_main:
                    try: default_opt_idx_main = options_list_main.index(previous_ans_val_main)
                    except ValueError: pass
                
                user_choice_active_main = st.radio(
                    "Sélectionnez votre réponse:", options=options_list_main, index=default_opt_idx_main,
                    key=f"q_radio_main_{current_q_idx_active_main}",
                    disabled=st.session_state.feedback_shown.get(current_q_idx_active_main, False)
                )

                submit_btn_pressed_main = st.button("Soumettre la réponse", key=f"submit_btn_main_{current_q_idx_active_main}",
                                               disabled=st.session_state.feedback_shown.get(current_q_idx_active_main, False))

                if submit_btn_pressed_main:
                    st.session_state.answers[current_q_idx_active_main] = user_choice_active_main
                    st.session_state.feedback_shown[current_q_idx_active_main] = True
                    st.rerun()

                if st.session_state.feedback_shown.get(current_q_idx_active_main, False):
                    stored_ans_active_main = st.session_state.answers.get(current_q_idx_active_main)
                    if stored_ans_active_main == correct_med_name_active_main:
                        st.success("Correct! ✅")
                    else:
                        st.error(f"Incorrect! ❌ La bonne réponse est : **{correct_med_name_active_main}**")
                    st.markdown(f"💡 Mnémonique: {processed_mnemonic_active_main}", unsafe_allow_html=False)
            
            st.markdown("---")
            nav_prev_col_main, nav_next_col_main = st.columns(2)
            with nav_prev_col_main:
                if st.button("⬅️ Précédent", disabled=current_q_idx_active_main <= 0, use_container_width=True):
                    st.session_state.question_index -= 1
                    st.rerun()
            with nav_next_col_main:
                next_btn_disabled_main = current_q_idx_active_main >= total_questions_active_main - 1 or \
                                    not st.session_state.feedback_shown.get(current_q_idx_active_main, False)
                if st.button("Suivant ➡️", disabled=next_btn_disabled_main, use_container_width=True):
                    st.session_state.question_index += 1
                    st.rerun()
            
            st.markdown("---")
            st.write("**Aller à la question :**")
            num_quick_nav_cols_main = min(total_questions_active_main, 10)
            quick_nav_cols_main = st.columns(num_quick_nav_cols_main)
            for i_nav_main in range(total_questions_active_main):
                col_for_nav_btn_main = quick_nav_cols_main[i_nav_main % num_quick_nav_cols_main]
                nav_label_main = str(i_nav_main + 1)
                nav_icon_main = ""
                if i_nav_main in st.session_state.answers:
                    is_correct_nav_q_main = st.session_state.answers[i_nav_main] == df_current_quiz_active_main.iloc[i_nav_main]['MedicationName']
                    nav_icon_main = " ✅" if is_correct_nav_q_main else " ❌"
                
                nav_btn_type_main = "primary" if i_nav_main == current_q_idx_active_main else "secondary"
                if col_for_nav_btn_main.button(f"{nav_label_main}{nav_icon_main}", key=f"quick_nav_main_{i_nav_main}", type=nav_btn_type_main, use_container_width=True):
                    st.session_state.question_index = i_nav_main
                    st.rerun()

            st.markdown("---")
            all_q_attempted_main = all(st.session_state.feedback_shown.get(i_att, False) for i_att in range(total_questions_active_main))
            if st.button("🏁 Terminer le Quiz et Voir les Résultats", use_container_width=True, disabled=not all_q_attempted_main):
                st.session_state.show_result = True
                st.rerun()
        else:
            st.warning("Index de question invalide. Redémarrage de la sélection du quiz.")
            st.session_state.question_index = 0
            st.session_state.quiz_loaded = False
            st.rerun()

    elif st.session_state.show_result:
        st.subheader("📊 Résultats du Quiz")
        df_quiz_review_main = st.session_state.current_quiz_df
        total_q_review_main = len(df_quiz_review_main)
        answers_review_main = st.session_state.answers
        correct_count_review_main = 0
        
        for i_rev_main in range(total_q_review_main):
            if i_rev_main in answers_review_main and answers_review_main[i_rev_main] == df_quiz_review_main.iloc[i_rev_main]['MedicationName']:
                correct_count_review_main += 1
        
        incorrect_count_review_main = len([ans_rev for i_rev, ans_rev in answers_review_main.items() if i_rev < total_q_review_main and ans_rev != df_quiz_review_main.iloc[i_rev]['MedicationName']])
        unanswered_review_main = total_q_review_main - len([i_rev for i_rev in answers_review_main.keys() if i_rev < total_q_review_main])


        stat_col1_main, stat_col2_main, stat_col3_main, stat_col4_main = st.columns(4)
        stat_col1_main.metric("✅ Correctes", correct_count_review_main)
        stat_col2_main.metric("❌ Incorrectes", incorrect_count_review_main)
        stat_col3_main.metric("❓ Non répondues", unanswered_review_main)
        score_val_main = (correct_count_review_main / total_q_review_main * 100) if total_q_review_main > 0 else 0
        stat_col4_main.metric("🏆 Score", f"{correct_count_review_main}/{total_q_review_main}", f"{score_val_main:.1f}%")

        res_btn_col_1_main, res_btn_col_2_main = st.columns(2)
        with res_btn_col_1_main:
            if st.button("🚀 Recommencer ce Quiz", use_container_width=True):
                current_section_data_main = quiz_sections_dict_main.get(st.session_state.selected_quiz_section_name)
                if current_section_data_main is not None:
                    st.session_state.current_quiz_df = current_section_data_main.copy().sample(frac=1).reset_index(drop=True)
                st.session_state.update({
                    'question_index': 0, 'answers': {}, 'feedback_shown': {},
                    'show_result': False, 'current_question_options': {}, 'quiz_loaded': True
                })
                st.rerun()
        with res_btn_col_2_main:
            if st.button("⚙️ Changer de Section", use_container_width=True):
                st.session_state.update({
                    'question_index': 0, 'answers': {}, 'feedback_shown': {},
                    'show_result': False, 'current_quiz_df': pd.DataFrame(),
                    'current_question_options': {}, 'quiz_loaded': False
                })
                st.rerun()

        with st.expander("🧐 Revoir Vos Réponses", expanded=False):
            if total_q_review_main == 0: st.write("Aucune question n'a été chargée pour la révision.")
            else:
                for i_detail_main in range(total_q_review_main):
                    q_data_detail_main = df_quiz_review_main.iloc[i_detail_main]
                    user_ans_detail_main = answers_review_main.get(i_detail_main, "Non Répondu")
                    correct_ans_detail_main = q_data_detail_main['MedicationName']
                    is_correct_detail_main = user_ans_detail_main == correct_ans_detail_main
                    raw_mnemonic_detail_main = q_data_detail_main.get('MnemonicText', "Pas de mnémonique disponible.")
                    processed_mnemonic_detail_main = re.sub(r'##(.*?)##', r'**\1**', raw_mnemonic_detail_main)
                    indication_detail_main = q_data_detail_main['IndicationFrench']
                    status_icon_detail_main = "❓"
                    if i_detail_main in answers_review_main: status_icon_detail_main = "✅" if is_correct_detail_main else "❌"

                    st.markdown(f"**Question {i_detail_main+1}:** Quel médicament est utile en cas de **{indication_detail_main}** ?")
                    
                    review_img_cat_detail_main = q_data_detail_main.get('Category', "Generic Drugs")
                    review_med_name_detail_main = q_data_detail_main['MedicationName']
                    review_img_url_detail_main = get_image_url(df_images, review_img_cat_detail_main, review_med_name_detail_main)
                    
                    rev_detail_left_main, rev_detail_right_main = st.columns([3,1])
                    with rev_detail_right_main:
                         display_image(review_img_url_detail_main, st)
                    with rev_detail_left_main:
                        st.write(f"Votre réponse : **{user_ans_detail_main}** {status_icon_detail_main}")
                        if not is_correct_detail_main and i_detail_main in answers_review_main:
                            st.write(f"Bonne réponse : **{correct_ans_detail_main}**")
                        elif user_ans_detail_main == "Non Répondu":
                            st.write(f"Bonne réponse : **{correct_ans_detail_main}**")
                        st.markdown(f"💡 Mnémonique: {processed_mnemonic_detail_main}", unsafe_allow_html=False)
                    st.divider()

    elif not st.session_state.quiz_loaded:
        st.info("👋 Bienvenue ! Veuillez sélectionner une section de quiz dans la barre latérale, puis cliquez sur 'Charger / Recommencer le Quiz'.")
