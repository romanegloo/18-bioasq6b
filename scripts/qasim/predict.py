import logging, coloredlogs
import code
import os
from pathlib import PosixPath

from BioAsq6B.QaSimSent import Predictor

# Initialize logger
logger = logging.getLogger()
coloredlogs.install(level='DEBUG',
                    fmt="[%(asctime)s %(levelname)s] %(message)s")

banner = """
usage:
  >>> usage()
  [prints out this banner]

  >>> p = Predictor("path_to_pretrained_model")
  [re-instantiate a predictor with the given pretained model]

  >>> scores, sentneces = p.get_qasim_scores(
        "Orteronel was developed for treatment of which cancer?",
        "factoid", 
        "Orteronel (also known as TAK-700) is a novel hormonal therapy that is "
        "currently in testing for the treatment of prostate cancer. Another "
        "dummy sentence here.")
  [prints out the QAsim scores by sentneces]
  note that 2nd parameter is a questions types: 
    ['yesno', 'factoid', 'list', 'summary']

  >>> scores, sentences = p.get_qasim_scores(ex_q, ex_qtype, ex_doc_rel)
  [get the scores of an example question and document content] 
"""

ex_q = "Orteronel was developed for treatment of which cancer?"
# ex_q = "Describe what is athelia syndrome?"
# ex_q = "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
ex_qtype = "summary"
ex_doc_rel = \
    """Orteronel (also known as TAK-700) is a novel hormonal therapy that is
     currently in testing for the treatment of prostate cancer. Orteronel inhibits
     the 17,20 lyase activity of the enzyme CYP17A1, which is important for
     androgen synthesis in the testes, adrenal glands and prostate cancer cells.
     Preclinical studies demonstrate that orteronel treatment suppresses androgen
     levels and causes shrinkage of androgen-dependent organs, such as the prostate
     gland. Early reports of clinical studies demonstrate that orteronel treatment
     leads to reduced prostate-specific antigen levels, a marker of prostate cancer
     tumor burden, and more complete suppression of androgen synthesis than
     conventional androgen deprivation therapies that act in the testes alone.
     Treatment with single-agent orteronel has been well tolerated with fatigue as
     the most common adverse event, while febrile neutropenia was the dose-limiting
     toxicity in a combination study of orteronel with docetaxel. Recently, the
     ELM-PC5 Phase III clinical trial in patients with advanced-stage prostate
     cancer who had received prior docetaxel was unblinded as the overall survival
     primary end point was not achieved. However, additional Phase III orteronel
     trials are ongoing in men with earlier stages of prostate cancer.
    """
# ex_doc_rel = """Hirschsprung disease (HSCR), or congenital intestinal
# aganglionosis, is a common hereditary disorder causing intestinal obstruction,
# thereby showing considerable phenotypic variation in conjunction with complex
# inheritance. Moreover, phenotypic assessment of the disease has been
# complicated since a subset of the observed mutations is also associated with
# several additional syndromic anomalies. Coding sequence mutations in e.g. RET,
# GDNF, EDNRB, EDN3, and SOX10 lead to long-segment (L-HSCR) as well as syndromic
# HSCR but fail to explain the transmission of the much more common short-segment
# form (S-HSCR). Furthermore, mutations in the RET gene are responsible for
# approximately half of the familial and some sporadic cases, strongly
# suggesting, on the one hand, the importance of non-coding variations and, on
# the other hand, that additional genes involved in the development of the
# enteric nervous system still await their discovery. For almost all of the
# identified HSCR genes incomplete penetrance of the HSCR phenotype has been
# reported, probably due to modifier loci. Therefore, HSCR has become a model for
# a complex oligo-/polygenic disorder in which the relationship between different
# genes creating a non-mendelian inheritance pattern still remains to be
# elucidated. """
# ex_doc_rel = """Athelia is a very rare entity that is defined by the absence of the nipple-areola complex. It can affect either sex and is mostly part of syndromes including other congenital or ectodermal anomalies, such as limb-mammary syndrome, scalp-ear-nipple syndrome, or ectodermal dysplasias. Here, we report on three children from two branches of an extended consanguineous Israeli Arab family, a girl and two boys, who presented with a spectrum of nipple anomalies ranging from unilateral hypothelia to bilateral athelia but no other consistently associated anomalies except a characteristic eyebrow shape. Using homozygosity mapping after single nucleotide polymorphism (SNP) array genotyping and candidate gene sequencing we identified a homozygous frameshift mutation in PTPRF as the likely cause of nipple anomalies in this family. PTPRF encodes a receptor-type protein phosphatase that localizes to adherens junctions and may be involved in the regulation of epithelial cell-cell contacts, peptide growth factor signaling, and the canonical Wnt pathway. Together with previous reports on female mutant Ptprf mice, which have a lactation defect, and disruption of one allele of PTPRF by a balanced translocation in a woman with amastia, our results indicate a key role for PTPRF in the development of the nipple-areola region."""
ex_doc_irrel = \
    """The dynamics of antibody response in guinea pigs infected with Coxiella
     burnetii was investigated by microagglutination (MA) and complement-fixation
     (CF) tests with different preparations of C. burnetii antigens. At the onset
     of antibody response the highest antibody titres were detected by the MA test
     with natural antigen 2, later on by the MA test with artificial antigen 2.
     Throughtout the 1-year period of observation, the CF antibody levels were
     usually lower and, with the exception of the highest infectious doses, the CF
     antibodies appeared later than agglutinating antibodies. There was no
     difference in the appearance of agglutinating and CF antibodies directed to
     antigen 1. Inactivation of the sera caused a marked decrease in antibody
     titres when tested with artificial antigen 2, whereas the antibody levels
     remained unchanged when tested with natural antigen 2.
    """
root_dir = PosixPath(__file__).absolute().parents[3].as_posix()
model_file = os.path.join(root_dir,
                          'data/qa_prox/var/20181215-ae25499d-best-acc93.mdl')
idf_file = os.path.join(root_dir, 'data/idf.pkl')
p = Predictor(model_file, load_wd=True)

code.interact(banner, local=locals())
