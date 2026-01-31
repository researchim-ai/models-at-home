import streamlit as st
import math

# Internationalization (i18n)
try:
    from homellm.i18n import t
except ImportError:
    from ..i18n import t

def render_docs():
    """Renders built-in documentation."""
    
    st.header(f"📚 {t('docs.title')}")
    st.markdown(t("docs.welcome"))

    # Create tabs inside documentation for convenient navigation
    doc_tab1, doc_tab2, doc_tab3, doc_tab4, doc_tab5 = st.tabs([
        f"🚀 {t('docs.quick_start')}", 
        f"📈 {t('docs.scaling_laws')}",
        f"🧠 {t('docs.how_it_works')}", 
        f"📖 {t('docs.terminology')}", 
        f"💾 {t('docs.data_preparation')}"
    ])

    # --- 1. STEP-BY-STEP GUIDE (Pretrain + SFT) ---
    with doc_tab1:
        st.markdown(t("docs.guide_content"))

    # --- 2. SCALING LAWS ---
    with doc_tab2:
        st.markdown(t("docs.scaling_intro"))
        
        # --- INTERACTIVE CALCULATOR ---
        st.markdown(f"### 🧮 {t('docs.calc_title')}")
        
        calc_options = [
            f"📊 {t('docs.calc_data_for_model')}", 
            f"🎯 {t('docs.calc_model_for_data')}",
            f"⚡ {t('docs.calc_compute')}"
        ]
        calc_mode = st.radio(
            t("docs.calc_question"),
            calc_options,
            horizontal=True
        )
        
        st.markdown("---")
        
        if calc_mode == calc_options[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                model_size = st.select_slider(
                    f"🧠 {t('docs.calc_model_size')}",
                    options=[25_000_000, 50_000_000, 100_000_000, 200_000_000, 
                            300_000_000, 500_000_000, 800_000_000, 1_000_000_000,
                            3_000_000_000, 7_000_000_000],
                    value=100_000_000,
                    format_func=lambda x: f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B"
                )
                
                tokens_per_param = st.slider(
                    f"📚 {t('docs.calc_tokens_per_param')}",
                    min_value=5,
                    max_value=100,
                    value=20,
                    help=t("docs.calc_tokens_help")
                )
            
            with col2:
                optimal_tokens = model_size * tokens_per_param
                compute_flops = 6 * model_size * optimal_tokens
                
                # Peak FP16/BF16 Tensor Core TFLOPS (без sparsity)
                gpu_options = {
                    "🎮 RTX 3090 (71 TFLOPS)": 71e12,
                    "🎮 RTX 4090 (165 TFLOPS)": 165e12,
                    "🎮 RTX 5090 (210 TFLOPS)": 210e12,
                    "🖥️ A100 80GB (312 TFLOPS)": 312e12,
                    "🖥️ H100 SXM (990 TFLOPS)": 990e12,
                    "🖥️ H100 PCIe (757 TFLOPS)": 757e12,
                    "🖥️ H200 SXM (990 TFLOPS)": 990e12,
                    "🖥️ H200 NVL (836 TFLOPS)": 836e12,
                }
                
                st.metric(f"📊 {t('docs.calc_tokens_needed')}", f"{optimal_tokens/1e9:.2f}B")
                st.metric(f"⚡ {t('docs.calc_compute_flops')}", f"{compute_flops:.2e}")
                
                # Rough estimate of data size (1 token ≈ 4 chars ≈ 4 bytes)
                data_size_gb = (optimal_tokens * 4) / 1e9
                st.metric(f"💾 {t('docs.calc_data_size')}", f"{data_size_gb:.1f} GB")
            
            # Time estimate
            st.markdown(f"#### ⏱️ {t('docs.calc_time_estimate')}")
            
            selected_gpu = st.selectbox(f"{t('docs.calc_select_gpu')}:", list(gpu_options.keys()))
            mfu = st.slider("MFU (Model FLOPS Utilization)", 0.1, 0.6, 0.3, 
                           help=t("docs.calc_mfu_help"))
            
            gpu_tflops = gpu_options[selected_gpu]
            effective_tflops = gpu_tflops * mfu
            training_seconds = compute_flops / effective_tflops
            training_hours = training_seconds / 3600
            training_days = training_hours / 24
            
            if training_days >= 1:
                st.info(f"⏱️ {t('docs.calc_approx')} **{training_days:.1f} {t('docs.calc_days')}** {t('docs.calc_on')} 1× {selected_gpu.split('(')[0].strip()}")
            else:
                st.info(f"⏱️ {t('docs.calc_approx')} **{training_hours:.1f} {t('docs.calc_hours')}** {t('docs.calc_on')} 1× {selected_gpu.split('(')[0].strip()}")
        
        elif calc_mode == calc_options[1]:
            col1, col2 = st.columns(2)
            
            with col1:
                data_tokens = st.number_input(
                    f"📚 {t('docs.calc_how_many_tokens')}",
                    min_value=1_000_000,
                    max_value=1_000_000_000_000,
                    value=1_000_000_000,
                    step=100_000_000,
                    format="%d"
                )
                st.caption(f"= {data_tokens/1e9:.2f}B tokens")
                
                tokens_per_param = st.slider(
                    f"📚 {t('docs.calc_tokens_per_param')}",
                    min_value=5,
                    max_value=100,
                    value=20,
                    key="tpp2"
                )
            
            with col2:
                optimal_params = data_tokens / tokens_per_param
                
                if optimal_params >= 1e9:
                    size_str = f"{optimal_params/1e9:.2f}B"
                else:
                    size_str = f"{optimal_params/1e6:.0f}M"
                
                st.metric(f"🧠 {t('docs.calc_optimal_size')}", size_str)
                
                # Recommendation from our list
                model_sizes = [
                    (25e6, "Tiny (25M)", t("docs.model_tiny_desc")),
                    (50e6, "Small (50M)", t("docs.model_small_desc")),
                    (100e6, "Base (100M)", t("docs.model_base_desc")),
                    (200e6, "Medium (200M)", t("docs.model_medium_desc")),
                    (300e6, "Large (300M)", t("docs.model_large_desc")),
                    (500e6, "XL (500M)", t("docs.model_xl_desc")),
                    (800e6, "XXL (800M)", t("docs.model_xxl_desc")),
                    (1e9, "1B", t("docs.model_1b_desc")),
                ]
                
                recommended = None
                for size, name, desc in model_sizes:
                    if optimal_params >= size:
                        recommended = (name, desc)
                
                if recommended:
                    st.success(f"✅ {t('docs.calc_recommend')}: **{recommended[0]}** — {recommended[1]}")
        
        else:  # Compute needed
            col1, col2 = st.columns(2)
            
            with col1:
                model_size = st.select_slider(
                    f"🧠 {t('docs.calc_model_size')}",
                    options=[25_000_000, 50_000_000, 100_000_000, 200_000_000, 
                            300_000_000, 500_000_000, 800_000_000, 1_000_000_000],
                    value=100_000_000,
                    format_func=lambda x: f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B",
                    key="ms2"
                )
                
                data_tokens = st.number_input(
                    f"📚 {t('docs.calc_tokens_for_training')}",
                    min_value=100_000_000,
                    max_value=100_000_000_000,
                    value=2_000_000_000,
                    step=500_000_000,
                    key="dt2"
                )
            
            with col2:
                compute_flops = 6 * model_size * data_tokens
                
                st.metric(f"⚡ {t('docs.calc_compute_flops')}", f"{compute_flops:.2e}")
                st.metric("📊 PetaFLOPs", f"{compute_flops/1e15:.2f}")
                
                # GPU-days (A100 = 312 TFLOPS, 30% MFU)
                a100_effective = 312e12 * 0.3
                gpu_days = compute_flops / a100_effective / 86400
                st.metric(f"🖥️ {t('docs.calc_gpu_days')}", f"{gpu_days:.1f}")
        
        st.markdown("---")
        
        # --- Model sizes table and conclusions ---
        st.markdown(t("docs.scaling_tables"))

    # --- 3. HOW IT WORKS ---
    with doc_tab3:
        st.markdown(t("docs.how_it_works_content"))

    # --- 4. TERMINOLOGY ---
    with doc_tab4:
        st.markdown(t("docs.terminology_content"))

    # --- 5. DATA PREPARATION ---
    with doc_tab5:
        st.markdown(t("docs.data_prep_content"))
