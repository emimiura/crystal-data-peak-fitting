import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
from time import sleep
from stqdm import stqdm
from datetime import datetime, timedelta
from lmfit import CompositeModel, Model
from lmfit.models import LinearModel, ExponentialModel, PseudoVoigtModel, GaussianModel, LorentzianModel

##### PAGE TITLE #####
st.set_page_config(page_title='Powder X-ray Diffraction Data Processing', page_icon='💎')
st.title('💎 Powder X-ray Diffraction Data Processing')
st.info('**This app is designed to perform peak fitting and analysis of powder X-ray diffraction data of nanocrystalline materials.**\n - The Scherrer Equation is used in conjunction with fitted peak positions and full width half maximum (FWHM) values to determine the approximate nanocrystalline domain size from X-ray diffraction spectra. \n - Bulk materials can also be analyzed for peak fitting, but the Scherrer Equation (diameter sizing equation) is not accurate for materials that are above ~50 nm in diameter.')
st.info('**How to use this app:**\n1. Please upload one `.csv` or `.xy` file containing two columns (2theta, intensity) of data separated by tabs. Raw `.xy` data exported from the DIFFRAC.EVA is uploadable.\n 2. Use the sidebar to customize your data proccesing and fit parameters. \n 3. Use fitted data results and sidebar to customize Scherrer Equation analysis of your fitted data.''')

##### SESSION STATE #####
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def set_state(i):
    st.session_state.stage = i


##### GENERATE / VISUALIZE DATA #####
generate_data_c = st.container(border=False)
## loading csv
generate_data_c.subheader('UPLOADED DATA')

st.sidebar.subheader('UPLOAD')
spectra = st.sidebar.file_uploader('upload file', type={'csv', 'xy'}, label_visibility='hidden')

col1, col2 = generate_data_c.columns([1,3])

if spectra is not None:
        # reading csv - old (csv only)
        # data_df = pd.read_csv(spectra, sep='\t', names=['2theta', 'intensity']) # old code
    with col1:
        # reading csv and/or XY data with text
        data_df = pd.read_csv(spectra, delimiter='\s+')
        data_df = data_df.apply(pd.to_numeric, errors='coerce')
        data_df = data_df.dropna(axis=1)
        data_df.columns = ['2theta', 'intensity']
        st.markdown('**Raw data**')
        st.write(data_df)

    with col2:
        # plotting data
        st.markdown('**Raw XRD data**')
        st.line_chart(data=data_df, x='2theta', y='intensity')

else: 
    generate_data_c.error('No data has been uploaded yet!!')



##### DATA PROCESSING #####
data_process_c = st.container(border=False)
data_process_c.subheader('PROCESSED DATA')
st.sidebar.subheader('DATA PROCESSING')

## decision for data processing 
process_choice = st.sidebar.radio('Choose data processing option:', 
                  ['None', 'Truncate data'],
                  captions = ['Use raw data for peak fitting.', 'Cut beginning and end of data.'],
                  horizontal = True)

if process_choice == 'None':
    data_process_c.write('You chose to use the raw data for peak fitting.')

    # for when data_df is undefined bc it hasn't been uploaded yet
    if spectra is not None:
        data = data_df
    else:
        pass
else:
    data_process_c.write('You chose to truncate the raw data before peak fitting.')

    st.sidebar.markdown('Chose a 2theta range to cut raw data:')
    st.sidebar.info('This choice will crop data to remove data below the lower bound and data above the upper bound. Lower Bound **MUST** be a lower 2theta value than Upper Bound. Truncated data is easier to baseline when the upper and lower bound terminate at the background intensity.')
    data_length = len(data_df)-1

    cola, colb = st.columns(2)
    with cola:
        lower_cut = st.sidebar.number_input(label='Lower Bound, 2theta', 
                                    min_value=data_df['2theta'][0], 
                                    max_value=data_df['2theta'][data_length],
                                    step=0.01)
    with colb:
        upper_cut = st.sidebar.number_input(label='Upper Bound, 2theta', 
                                    min_value=data_df['2theta'][0], 
                                    max_value=data_df['2theta'][data_length],
                                    step=0.01)

    trunc_data = data_df.drop(data_df.loc[(data_df['2theta']<lower_cut) | (data_df['2theta']>upper_cut)].index)
    data_process_c.markdown('Truncated data:')
    data_process_c.line_chart(data=trunc_data, x='2theta', y='intensity')
    data = trunc_data


## confirmation of data post-processing
if spectra is not None:
    data_process_c.success('Just to confirm, the following data is what will be fit:')
    data_process_c.line_chart(data=data, x='2theta', y='intensity')
else:
    data_process_c.error('No data to see here 👀')


##### MODEL / PEAK DECISIONS #####
fitting_data_c = st.container(border=False)

fitting_data_c.subheader('FITTING DATA')
st.sidebar.subheader('FITTING DATA')
## identify number of peaks
st.sidebar.info("Chose the number of peaks you'd like to fit and the initial guesses for peak positions as integer values.")
peak_num = st.sidebar.slider(label='How many peaks would you like to fit?',
                     min_value=1,
                     max_value=6,
                     on_change=set_state,
                     args=[0])

## peak positions
fitting_data_c.markdown(f'Number of peaks: `{peak_num}`', unsafe_allow_html=True)

parameters = []
for i in range(peak_num):
    x = st.sidebar.number_input(label=f"Peak position {i+1}: (p{i+1}_)", 
                                value=i+1, 
                                key=None, 
                                on_change=set_state,
                                args=[0])
    parameters.append(x)

fitting_data_c.info('Suggestions on chosing a model to fit your data:\n 1. Pseudo-Voigt models are used for traditional looking XRD data. \n 2. Play around with mixing and matching the peak fitting model and baseline fitting model to maximize the R$$^2$$.')

col1, col2 = fitting_data_c.columns(2)
with col1:
    ## choose your model
    model_choice = st.radio('Choose peak fitting model:', 
                            ['Pseudo-Voigt', 'Gaussian', 'Lorentzian'],
                            captions=['Model traditionally used for XRD data.', 'Model fit for broader peaks.', 'Model fit for narrower peaks.'],
                            on_change=set_state,
                            args=[0])

    if model_choice == 'Pseudo-Voigt':
        st.write('You chose to use the Pseudo-Voigt Model.')
        model = PseudoVoigtModel
        # prefix = 'pvoigt'
    elif model_choice == 'Gaussian':
        st.write('You chose to use the Gaussian Model.')
        model = GaussianModel
        # prefix = 'gauss'
    else:
        st.write('You chose to use the Lorentzian Model.')
        model = LorentzianModel
        # prefix = 'loren'
with col2:
    ## choose backgroud fit
    bkg_choice = st.radio('Choose baseline fitting model:',
                          ['Linear', 'Exponential'],
                          on_change=set_state,
                          args=[0])

    if bkg_choice == 'Linear':
        st.write('You chose to use the Linear baseline.')
        bkg = LinearModel(prefix='lin_')
        bkg.set_param_hint('intercept', value=0, vary=True)
        bkg.set_param_hint('slope', value=0, vary=True)
        bkg_pref = 'lin_'

    else:
        st.write('You chose to use the Exponential baseline.')
        bkg = ExponentialModel(prefix='exp_')
        #### add in exponential model background parameter hints?????
        bkg_pref = 'exp_'

# button to actually build and perform fit
if st.session_state.stage == 0:
    fitting_data_c.button('DO THE FIT!', on_click=set_state, args=[1])


if st.session_state.stage >= 1:
    reset_button = fitting_data_c.button('RESET', on_click=set_state, args=[0])

    for i in stqdm(range(100), desc='Loading your model...', backend=False, frontend=True):
        sleep(0.05)

    ##### BUILDING THE MODEL #####
    def build_model(num, model):
        pref = 'p{0}_'.format(num)
        model = model(prefix = pref)
        model.set_param_hint(pref+'center', value=parameters[num])
        return model

    mod = None
    for i in range(len(parameters)):
        add_mod = build_model(i, model)
        if mod is None:
            mod = add_mod
        else:
            mod = mod + add_mod

    ##### FITTING THE MODEL #####
    mod = mod + bkg

    result = mod.fit(data=data['intensity'].values, 
                x=data['2theta'].values, 
                method='leastsq')
    comps = result.eval_components(x=data['2theta'].values)
    dely = result.eval_uncertainty(sigma=3)
    # print(result.fit_report(min_correl=0.5))

    # calculating R^2
    r_sq = 1 - result.residual.var() / np.var(data['intensity'].values)

    ##### PLOTTING THE RESULT #####

    def plot_model(data, result, comps, dely):
        # resetting truncated data index
        data.reset_index(drop=True, inplace=True)
        
        # making df out of result components (n peaks + bkg)
        comps_df = pd.DataFrame(comps)
        comps_df.reset_index(drop=True, inplace=True)

        # total dataset with raw data, 
        total_data = pd.concat([data, comps_df], axis=1)
        total_data['best_fit'] = pd.Series(result.best_fit)
        
        prefs = ['intensity', 'best_fit', bkg_pref]
        for i in range(len(parameters)):
            pref = 'p{0}_'.format(i)
            prefs.append(pref)
        
        fitting_data_c.markdown('Fitted data set plot:')
        # chart_data = (
        #     alt.Chart(total)
        # )
        #meh streamlit chart, need to udpate with altair chart
        fitting_data_c.line_chart(data=total_data, x='2theta', y=prefs)

        total_data['residual'] = total_data['intensity'] - total_data['best_fit']
        fitting_data_c.markdown('Residual fit plot:')
        fitting_data_c.line_chart(data=total_data, x='2theta', y=['intensity','residual', 'best_fit'])

        return total_data

    # plotting data with raw data, best fit, individual components, and background fit
    total_data = plot_model(data, result, comps, dely)
else:
    result = None
    total_data = None

## EXTRACTING FIT RESULTS ##
if result is not None:
    def extract_results(result):
        result_centers = []
        result_fwhms = []
        peak_results = pd.DataFrame()

        prefs = []
        for i in range(len(parameters)):
            pref = 'p{0}_'.format(i)
            prefs.append(pref)

        for i in range(0,len(prefs)):
            center = result.params[f'{prefs[i]}center'].value
            # print(center) #testing
            result_centers.append(center)
            fwhm = result.params[f'{prefs[i]}fwhm'].value
            result_fwhms.append(fwhm)
            # print(fwhm) #testing

        peak_results['peak_centers (2theta)'] = pd.Series(result_centers)
        peak_results['fwhms'] = pd.Series(result_fwhms)

        return peak_results

    peak_results = extract_results(result)
else:
    pass

# downloading data

if total_data is not None:
    @st.cache_data
    def convert_df_to_csv(total_data):
        return total_data.to_csv().encode('utf-8')
    result_csv = convert_df_to_csv(total_data)
    peak_result_csv = convert_df_to_csv(peak_results)

        # Fit report
    fit_report_c = st.container(border=False)
    fit_report_c.subheader('FIT RESULTS')
    with fit_report_c.expander('**Fit Results**'):
        st.download_button(label='DOWNLOAD FIT RESULTS (.CSV)',
                        data=peak_result_csv,
                        file_name='CrystDataPeakFitApp_FitResults.csv',
                        mime='text/csv')
        st.markdown(f'R$$^2$$: `{r_sq:.5f}`', unsafe_allow_html=True)
        st.write(peak_results)
    with fit_report_c.expander('**Fitted Dataset**'):
        st.download_button(label='DOWNLOAD DATASET (.CSV)',
                                data=result_csv,
                                file_name='CrystDataPeakFitApp_FittedData.csv',
                                mime='text/csv')
        st.write(total_data)
    with fit_report_c.expander('**Full Fit Report**'):
        st.code(result.fit_report(min_correl=0.9))
else:
    pass

