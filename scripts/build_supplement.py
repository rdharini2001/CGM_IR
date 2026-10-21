"""Typeset the extended scholarly methods for the associated article.

All results are drawn from model_scores.csv, observed_30.csv and hba1c.csv.
No printed article summary metric is imported into this document's fitted results.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, KeepTogether, PageBreak, HRFlowable)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs';ASSETS=DOCS/'assets';ASSETS.mkdir(exist_ok=True)
OBS=pd.read_csv(ROOT/'data/observed_30.csv')
HBA=pd.read_csv(ROOT/'data/hba1c.csv')
RESULTS=pd.read_csv(ROOT/'results/model_scores.csv')

normal='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
if Path(normal).exists():
    pdfmetrics.registerFont(TTFont('OpticReg',normal))
    pdfmetrics.registerFont(TTFont('OpticBold',bold))
    FONT,BOLD='OpticReg','OpticBold'
else: FONT,BOLD='Helvetica','Helvetica-Bold'
INK=colors.HexColor('#0a2630');GREEN=colors.HexColor('#23756a');COPPER=colors.HexColor('#b6774a')
MUTED=colors.HexColor('#526862');LIGHT=colors.HexColor('#ebf4f1');RULE=colors.HexColor('#dce9e5')
styles={
 'kicker':ParagraphStyle('kicker',fontName=BOLD,fontSize=8.2,leading=11,spaceAfter=9,textColor=GREEN),
 'title':ParagraphStyle('title',fontName=BOLD,fontSize=19.0,leading=25,spaceAfter=12,textColor=INK),
 'subtitle':ParagraphStyle('subtitle',fontName=FONT,fontSize=9.3,leading=15,spaceAfter=17,textColor=MUTED),
 'h1':ParagraphStyle('h1',fontName=BOLD,fontSize=11.3,leading=16,spaceBefore=17,spaceAfter=8,textColor=GREEN,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName=BOLD,fontSize=9.2,leading=14,spaceBefore=9,spaceAfter=5,textColor=INK),
 'body':ParagraphStyle('body',fontName=FONT,fontSize=8.5,leading=14.0,spaceAfter=8,textColor=INK),
 'small':ParagraphStyle('small',fontName=FONT,fontSize=7.5,leading=11.8,spaceAfter=6,textColor=MUTED),
 'eq':ParagraphStyle('eq',fontName=FONT,fontSize=9.0,leading=14.8,spaceBefore=3,spaceAfter=10,leftIndent=14,textColor=GREEN),
 'headcell':ParagraphStyle('headcell',fontName=BOLD,fontSize=7.3,leading=10.5,textColor=colors.white),
 'cell':ParagraphStyle('cell',fontName=FONT,fontSize=7.2,leading=10.5,textColor=INK),
}
content=[]
def para(s,sty='body'): content.append(Paragraph(s,styles[sty]))
def h(s): para(s,'h1')
def eq(s):para(s,'eq')
def tab(head,rows,widths):
    cells=[[Paragraph(str(v),styles['headcell']) for v in head]]+[
        [Paragraph(str(v),styles['cell']) for v in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT',repeatRows=1)
    t.setStyle(TableStyle([
      ('BACKGROUND',(0,0),(-1,0),GREEN),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,LIGHT]),
      ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),('RIGHTPADDING',(0,0),(-1,-1),9),
      ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),8),
      ('LINEBELOW',(0,-1),(-1,-1),.4,RULE)]))
    content.append(t);content.append(Spacer(1,10))

def make_charts():
    fig,ax=plt.subplots(figsize=(9,3.0),facecolor='white');ax.set_facecolor('white')
    for subject,group in OBS.groupby('subject',sort=True):
        q=group.set_index('phase').loc[['before','after','random']]
        ax.plot([0,1,2],q.reference_mgdl,color='#2b9586',lw=1.35,alpha=.43)
        ax.scatter([0,1,2],q.device_mgdl,color='#ca8e62',s=15,alpha=.76)
    ax.set_xticks([0,1,2],['Before','After','Random']);ax.set_ylabel('Glucose (mg/dL)');ax.grid(alpha=.13)
    for sp in ax.spines.values():sp.set_visible(False)
    fig.tight_layout();f=ASSETS/'supp_glucose_series.png';fig.savefig(f,dpi=165);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,2.8),facecolor='white')
    x=np.arange(len(HBA))
    ax.plot(x,HBA.actual_hba1c_percent,'o-',c='#2b9586',lw=1.6,label='Reference HbA1c')
    ax.plot(x,HBA.reported_estimated_hba1c_percent,'s--',c='#ca8e62',lw=1.5,label='Reported estimate')
    ax.set_xticks(x,HBA.subject);ax.set_ylabel('HbA1c (%)');ax.grid(alpha=.15);ax.legend(loc='upper left',frameon=False,fontsize=8,ncol=2)
    for sp in ax.spines.values():sp.set_visible(False)
    fig.tight_layout();g=ASSETS/'supp_hba1c_series.png';fig.savefig(g,dpi=165);plt.close(fig)
    return f,g
FIG1,FIG2=make_charts()

para('SUPPLEMENTARY METHODS  /  TECHRXIV 2023','kicker')
para('Near-Infrared Glucose Sensing and PSO-ANN Calibration','title')
para('Additional computational methods, statistical definitions and reproducible analysis for <i>Non-Invasive Continuous Glucose Monitoring using Near Infrared Sensors and PSO-ANN Algorithm</i> [1].','subtitle')
para('<b>Suma K. V. · Dharini Raghavan · Maya V. Karki · Narayana Sharma · Gundu H. R. Rao</b><br/>Article: doi:10.36227/techrxiv.24465955.v1  |  Code: github.com/rdharini2001/CGM_IR','small')
content.append(HRFlowable(width='100%',thickness=0.8,color=RULE,spaceBefore=6,spaceAfter=13))
para('<b>Summary.</b> The research article introduces a fingertip infrared glucose-sensing system and a particle-swarm-optimized neural calibration model. Here we describe the additional computational realization used to analyze the measurements printed in the article. The methods below specify input representation, the PSO network, a regularized residual extension, the noise-resolution assumption, evaluation populations and clinical-summary calculations. These details permit a complete rerun of the accompanying discovery-cohort calibration study. The terminology distinguishes calibration fit from transfer evaluation, consistent with infrared chemometrics practice [2-4].')

h('S1. Research measurement system and data organization')
para('We employ the 940 nm near-infrared emitter and photodetector architecture described in the article [1]. Optical transmission through the fingertip is received by the PT-333C photodetector, conditioned using filtering and amplifier stages, and converted to a voltage supplied to the calibration method. The response is reference glucose concentration in mg/dL. The neural inputs are the measured voltage, participant age and body mass index (BMI).')
para('The printed glucose-loading table provides ten participants with a before-loading, after-loading and random sampling record, yielding 30 voltage/reference pairs. A separate series of ten additional participants gives recorded glucose and HbA1c comparisons. We preserve their distinction: only the former supplies row-level input triples for neural calibration. Subject labels serve as grouping keys and are not included in the three-input model.')
para('In an ideal transmission measurement, optical attenuation may be expressed as A = log(I<sub>0</sub>/I), where I<sub>0</sub> and I are source and transmitted intensities. The recorded front-end voltage is a conditioned electronic response, not a directly measured absorbance. Therefore the code standardizes voltage, age and BMI within each fitting partition without asserting a one-to-one optical absorbance identity. Broader spectral preprocessing techniques provide relevant context but are not applied to this single-channel voltage series [10].')

h('S2. Neural architecture and swarm objective')
para('For two sigmoid hidden layers, we write the network output as a linear readout of the second hidden layer. The connection count for input dimension 3, hidden widths h<sub>1</sub> and h<sub>2</sub>, and one scalar output is 3h<sub>1</sub> + h<sub>1</sub>h<sub>2</sub> + h<sub>2</sub>. For the highlighted 3-3-2-1 architecture, this gives 9 + 6 + 2 = 17 trained connection weights. The eight network topologies listed in Table 1 are implemented using the same input features.')
eq('h<sub>1</sub> = sigmoid(W<sub>1</sub><sup>T</sup> z), &nbsp; h<sub>2</sub> = sigmoid(W<sub>2</sub><sup>T</sup>h<sub>1</sub>), &nbsp; c-hat = s<sub>y</sub>W<sub>3</sub><sup>T</sup>h<sub>2</sub> + m<sub>y</sub>.')
para('The particle vector concatenates the three weight matrices; particle movement follows the personal-best and global-best updates of Kennedy and Eberhart [5]. We use the study parameters of 25 particles, 70 updates, inertia coefficient 1.25, personal acceleration 2.76 and swarm acceleration 2.34 [1], together with bounded weight and velocity updates to maintain numerical stability.')
eq('v<sup>t+1</sup> = 1.25v<sup>t</sup> + 2.76r<sub>1</sub>(p - w<sup>t</sup>) + 2.34r<sub>2</sub>(g - w<sup>t</sup>); &nbsp; w<sup>t+1</sup> = w<sup>t</sup> + v<sup>t+1</sup>.')
para('The additional computational implementation uses standardized supervised squared loss, a small weight-size penalty and a local voltage-jitter penalty. The latter encourages a smooth calibrated response to a sub-millivolt perturbation of the instrument signal. These auxiliary terms are specified here as supplementary implementation choices, not as formulas printed in the main article.')
eq('J(w) = mean[(y - f<sub>w</sub>(x))<sup>2</sup>] + eta mean(w<sup>2</sup>) + xi mean[(f<sub>w</sub>(V+delta) - f<sub>w</sub>(V-delta))<sup>2</sup>].')

h('S3. Nonlinear residual calibration')
para('Because compact two-layer networks are constrained by their finite parameter count, we consider an additional smooth residual representation. This extension uses a positive-semidefinite Gaussian kernel and a ridge penalty; the representer theorem gives a finite expansion over the training observations [6]. Let r denote reference glucose minus the fitted neural estimate on the calibration records. The correction and its dual coefficients are')
eq('c-hat(z) = f<sub>w</sub>(z) + k(z,Z)<sup>T</sup> beta; &nbsp; beta = (K + alpha I)<sup>-1</sup>r, &nbsp; K<sub>ij</sub> = exp[-gamma ||z<sub>i</sub>-z<sub>j</sub>||<sup>2</sup>].')
para('We define the Gaussian scale using the median squared distance from each distinct observed configuration to its nearest nonidentical neighbor. This choice anchors the kernel to local geometric spacing rather than a reported manuscript error. The assumed glucose-reference quantization interval is Delta = 1 mg/dL; a uniform rounding model gives variance Delta<sup>2</sup>/12. We set alpha equal to this reference-resolution variance divided by the sample variance of the training reference values. This is a declared resolution model, not a measured estimate of full instrument or biological uncertainty.')
eq('gamma = ln(2) / median(nearest distinct squared distance); &nbsp; alpha = (Delta<sup>2</sup>/12) / Var<sub>train</sub>(y).')
para('As sensitivity analyses, the code also permits variance assumptions based on duplicated input configurations and generalized cross-validation. In the recorded series an identical voltage/age/BMI triple occurs at two different sampling phases; this is not a controlled replicate study, so the resulting dispersion is not interpreted as a validated sensor-noise estimate. GCV is a standard method for selecting smoothing strength from data [7], but its small-sample optimum need not transfer to unseen individuals.')

h('S4. Evaluation populations and metric definitions')
para('Three explicitly named statistical populations are retained. <b>Discovery-cohort calibration</b> scores the same 30 observations on which each model is fitted; it is an in-sample measure of calibration agreement. <b>Participant-held-out validation</b> uses five GroupKFold partitions in which all phases for one participant remain in the same fold. <b>Measurement-phase-held-out validation</b> withholds each sampling phase and retrains on the other phases, so participants may appear across training and evaluation. All scaling, swarm weights and kernel coefficients are refitted within the relevant training partition.')
para('IR chemometric practice supports the separation of calibration quality from cross-validated and independently predicted accuracy [2,3]. Studies of NIR glucose measurement also illustrate that individual calibration and its stability are independent concerns [4,11]. The code therefore reports all three evaluation settings with unique names; it does not assume which statistical partition produced every aggregate metric in the article.')
eq('MAE = mean|y - y-hat|; &nbsp; MSE = mean(y - y-hat)<sup>2</sup>; &nbsp; RMSE = sqrt(MSE); &nbsp; R<sup>2</sup> = 1 - SSE/SST.')
para('The supplementary adjusted coefficient of determination is calculated for the three input predictors as R<sup>2</sup><sub>adj</sub> = 1 - (1-R<sup>2</sup>)(n-1)/(n-3-1). For the eight architecture plots, we provide 100 R<sup>2</sup> as a clearly identified accuracy <i>proxy</i>; the article does not give an explicit formula for the architecture accuracy percentages. It is not asserted that those percentages and the proxy have identical definitions.')

h('S5. Calibration results on the printed 30-record discovery series')
rows=[]
for name in ['3-3-2-1','3-4-5-1','Linear Regression','Random Forest Regression','Artificial Neural Network','PSO-ANN']:
    x=RESULTS[(RESULTS.model==name)&(RESULTS.protocol=='discovery_fit')].iloc[0]
    rows.append([name,f'{x.mae:.3f}',f'{x.rmse:.3f}',f'{x.r2:.4f}'])
tab(['Network / regression family','MAE','RMSE','R-squared'],rows,[6.7*cm,2.3*cm,2.3*cm,3.3*cm])
para('Table S1. Calculated discovery-cohort fit on the 30 printed glucose-loading observations. These values use the standard definitions above, are recomputed from observed model residuals, and are not external prediction scores. The kernel correction applies across model families. Published manuscript score thresholds are not included in loss construction or hyperparameter selection.')
para('As a separate robustness check, the repository includes 45 model/protocol evaluations (15 model configurations times three populations). Those supplementary results are machine-readable in results/model_scores.csv and may be selected on the website without altering the core calibration models.')

h('S6. Measured instrument agreement and clinical summaries')
err=OBS.reference_mgdl.to_numpy()-OBS.device_mgdl.to_numpy()
para(f'The 30 recorded optical instrument outputs, compared directly against their tabulated invasive references, give mean absolute error {np.mean(abs(err)):.3f} mg/dL and root mean squared error {np.sqrt(np.mean(err**2)):.3f} mg/dL. They are physical instrument estimates already printed in the article, not fitted predictions from the software model. The displayed sample falls in Clarke Zone A using the conventional &lt;70 mg/dL absolute-tolerance and otherwise 20%-relative boundary [8]. The full clinical error-grid percentage quoted in the article must be kept distinct from the population enumerated in Table 3.')
content.append(Image(str(FIG1),width=16.3*cm,height=5.45*cm))
para('Figure S1. Reference series (green) and recorded non-invasive instrument values (copper) across the three measurement phases for participants S1-S10.','small')
para('We retain HbA1c records for participants S11-S20 as the second clinical series. For physiological interpretation, the ADAG relationship derived by Nathan et al. [9] uses a longitudinal average rather than a single spot-glucose value:')
eq('eAG (mg/dL) = 28.7 × HbA1c (%) - 46.7; &nbsp; HbA1c (%) = [eAG + 46.7]/28.7.')
content.append(Image(str(FIG2),width=16.3*cm,height=5.05*cm))
para('Figure S2. Reported HbA1c estimates and clinical reference HbA1c values for the separately tabulated subjects S11-S20. Values are reproduced from the article rather than recomputed from unavailable longitudinal histories.','small')

h('S7. Computational reproducibility and research scope')
para('The complete workflow uses Python, NumPy, SciPy, scikit-learn and joblib. After installation, bash run_all.sh fits the eight neural architectures and seven regression families, computes calibration and transfer results, exports all paired predictions, generates the interactive paper website and supplementary figures, and executes verification tests. Model-fitting modules depend on observed data only; manuscript table values are read solely by the separate post-fitting comparison audit.')
para('The calibrated system is studied as a retrospective research prototype. The discovery-cohort results characterize fit on the observed sample and are not represented as regulatory validation or clinical accuracy for new users. The measured instrument comparison, glucose loading responses and HbA1c observations remain distinct analyses, consistent with the focus of the associated article [1,8,9].')

h('References')
refs=[
'[1] Suma K. V., Raghavan D., Karki M. V., Sharma N., Rao G. H. R. Non-Invasive Continuous Glucose Monitoring using Near Infrared Sensors and PSO-ANN Algorithm. TechRxiv, 2023. doi:10.36227/techrxiv.24465955.v1.',
'[2] ASTM International. ASTM E1655-17(2024): Standard Practices for Infrared Multivariate Quantitative Analysis. store.astm.org/e1655-17r24.html.',
'[3] Near-infrared spectroscopy and hyperspectral imaging: non-destructive analysis of biological materials. Chemical Society Reviews, 2014. doi:10.1039/C4CS00062E.',
'[4] Sämann A. et al. Non-invasive blood glucose monitoring by means of near infrared spectroscopy: investigation of long-term accuracy and stability. Exp Clin Endocrinol Diabetes, 2000. doi:10.1055/s-2000-8137.',
'[5] Kennedy J., Eberhart R. Particle swarm optimization. Proceedings of ICNN, 1995. doi:10.1109/ICNN.1995.488968.',
'[6] Schölkopf B., Herbrich R., Smola A. J. A Generalized Representer Theorem. COLT, 2001. doi:10.1007/3-540-44581-1_27.',
'[7] Golub G. H., Heath M., Wahba G. Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter. Technometrics, 1979. doi:10.1080/00401706.1979.10489751.',
'[8] Clarke W. L. et al. Evaluating Clinical Accuracy of Systems for Self-Monitoring of Blood Glucose. Diabetes Care, 1987. doi:10.2337/diacare.10.5.622.',
'[9] Nathan D. M. et al. Translating the A1C Assay Into Estimated Average Glucose Values. Diabetes Care, 2008. doi:10.2337/dc08-0545.',
'[10] Rinnan Å., van den Berg F., Engelsen S. B. Review of the most common pre-processing techniques for near-infrared spectra. TrAC, 2009. doi:10.1016/j.trac.2009.07.007.',
'[11] Review of Non-Invasive Glucose Sensing Techniques: Optical, Electrical and Breath Acetone. Sensors, 2020. PMC7085605.'
]
for x in refs:para(escape(x),'small')

def draw_page(canvas,doc):
    canvas.saveState();w,h=A4
    canvas.setFillColor(INK);canvas.rect(0,h-13,w,13,stroke=0,fill=1)
    canvas.setFillColor(colors.HexColor('#80d9c6'));canvas.rect(43,h-16,46,3,stroke=0,fill=1)
    canvas.setFillColor(MUTED);canvas.setFont(FONT,7.1)
    canvas.drawString(42,28,'OPTIC  /  Supplementary Methods  /  Suma et al. (2023)')
    canvas.drawRightString(w-42,28,f'{doc.page}')
    canvas.restoreState()

p=DOCS/'SUPPLEMENTARY_METHODS.pdf'
doc=SimpleDocTemplate(str(p),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=44,bottomMargin=49,
 title='Supplementary Methods - Near-Infrared Glucose Sensing and PSO-ANN Calibration',
 author='Suma K. V.; Dharini Raghavan; Maya V. Karki; Narayana Sharma; Gundu H. R. Rao',
 subject='Additional computational calibration and evaluation methods associated with doi:10.36227/techrxiv.24465955.v1')
doc.build(content,onFirstPage=draw_page,onLaterPages=draw_page)
print('Created',p,p.stat().st_size)
