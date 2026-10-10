const fs = require("fs");
const D = require("docx");
const { Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, LevelFormat, PageBreak } = D;
const dims = JSON.parse(fs.readFileSync("results/img_dims.json"));
const FONT = "Times New Roman";
const W = 9026;

const t = (s, o = {}) => new TextRun({ text: s, font: FONT, size: 22, ...o });
const p = (runs, o = {}) => new Paragraph({ spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED,
  children: (typeof runs === "string" ? [t(runs)] : runs), ...o });
const h1 = s => new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 }, children: [new TextRun({ text: s, font: FONT, size: 28, bold: true })] });
const h2 = s => new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 }, children: [new TextRun({ text: s, font: FONT, size: 24, bold: true })] });
const bullet = (runs) => new Paragraph({ numbering: { reference: "b", level: 0 }, spacing: { after: 60, line: 276 }, children: typeof runs === "string" ? [t(runs)] : runs });
const caption = s => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [t(s, { italics: true, size: 20 })] });

function img(path, maxW = 560) {
  const [w, h] = dims[require("path").resolve(path)];
  const sc = Math.min(1, maxW / w);
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 100, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: Math.round(w * sc), height: Math.round(h * sc) },
      altText: { title: path, description: path, name: path } })] });
}

const bd = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
const borders = { top: bd, bottom: bd, left: bd, right: bd };
function table(cols, rows, widths, boldFirstCol = false) {
  const mk = (txt, w, head, i) => new TableCell({ borders, width: { size: w, type: WidthType.DXA },
    shading: head ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [t(String(txt), { size: 20, bold: head || (boldFirstCol && i === 0) })] })] });
  return new Table({ width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: cols.map((c, i) => mk(c, widths[i], true, i)) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => mk(c, widths[i], false, i)) }))] });
}
const tcap = s => new Paragraph({ spacing: { before: 160, after: 80 }, children: [t(s, { bold: true, size: 20 })] });
const gap = () => new Paragraph({ spacing: { after: 100 }, children: [] });

const kids = [];
// ---------------- Title
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
  children: [new TextRun({ text: "Design and Implementation of a Basic Information Retrieval System on the CISI Collection", font: FONT, size: 34, bold: true })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [t("Week 6 Assignment: Information Retrieval", { size: 24 })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [t("Student name: [your name]    Student ID: [your ID]", { size: 22 })] }));
kids.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: [t("GitHub repository: ", { size: 22, bold: true }), t("[paste your GitHub link here]", { size: 22 })] }));

// ---------------- 1 Intro
kids.push(h1("1. Introduction"));
kids.push(p("This report describes a basic ad-hoc information retrieval (IR) system that I designed and implemented in Python for the CISI collection. The system covers the full pipeline requested in the brief: document preprocessing, indexing, retrieval model selection and query processing. It was built from scratch (only NLTK's Porter stemmer is reused) so that every step can be explained and measured. The methods are taken from the chapters of Manning, Raghavan and Schütze (2009) that precede Chapter 13: tokenisation and normalisation (Chapter 2), inverted indexes (Chapters 1 and 4), the vector space model (Chapter 6), probabilistic retrieval and BM25 (Chapter 11), language models for IR (Chapter 12), relevance feedback (Chapter 9) and evaluation (Chapter 8)."));
kids.push(p("Chapter 13 itself concerns text classification and Naive Bayes, so it contributes ideas rather than components. Three of them shaped the work. First, Section 13.2.1 shows that multinomial Naive Bayes is formally identical to the multinomial unigram language model of Chapter 12, which motivated including a query-likelihood model as one of the three rankers. Second, Section 13.5 explains how removing uninformative terms reduces noise and vocabulary size, which I treat as the rationale for the stop-word and stemming experiments. Third, Section 13.6 insists on a strict separation of training, development and test data; I therefore tuned all parameters on one half of the queries and report results on the other half."));

// ---------------- 2 Dataset
kids.push(h1("2. Dataset"));
kids.push(p("CISI (Computer and Information Science Index) is a classic test collection of bibliographic records from library and information science, distributed in the SMART format. The three files used are CISI.ALL (documents), CISI.QRY (queries) and CISI.REL (relevance judgments). Each document has a title (.T), authors (.A), an abstract (.W) and cross-references (.X). I index the concatenation of title and abstract. Table 1 summarises the collection."));
kids.push(tcap("Table 1. CISI collection statistics"));
kids.push(table(["Property", "Value"], [
  ["Documents", "1,460"], ["Queries (total)", "112"], ["Queries with relevance judgments", "76"],
  ["Relevant (query, document) pairs", "3,114 (about 41 per judged query)"],
  ["Average document length (tokens, before stop-word removal)", "128.5"],
  ["Relevance scale", "Binary (listed = relevant)"]], [5200, 3826]));
kids.push(gap());
kids.push(p("Two properties matter for interpretation. Queries are long, natural-language information needs (often two or three sentences), and judged queries have about 41 relevant documents on average, around 2.8% of the collection. Consequently recall at 10 cannot exceed roughly 0.24 for a typical query, so low R@10 values are expected and precision, MAP and nDCG are the more informative measures. The 36 queries without judgments cannot be scored and are excluded from evaluation."));

// ---------------- 3 Design
kids.push(h1("3. System Design"));
kids.push(img("results/fig_architecture.png", 590));
kids.push(caption("Figure 1. System architecture: offline indexing and online query processing share one analyzer."));
kids.push(h2("3.1 Preprocessing (Chapter 2)"));
kids.push(p("Text is lowercased and split into alphanumeric tokens with a regular expression. A compact list of about 130 English function words is then removed, and the remaining tokens are reduced to stems with the Porter algorithm (Porter, 1980). Documents and queries pass through the same Preprocessor object, which guarantees that query terms match index terms. I evaluate three pipelines to quantify the contribution of each step: (A) tokenisation and lowercasing only, (B) A plus stop-word removal, (C) B plus stemming."));
kids.push(h2("3.2 Indexing (Chapters 1 and 4)"));
kids.push(p("The index is an in-memory inverted index mapping each term to a postings list of (docID, term frequency) pairs, ordered by docID. Because documents are processed in ascending docID order, postings remain sorted without a separate sorting pass. The index also stores document frequency, collection frequency, document lengths, the average document length and a forward index (docID to term counts). The forward index is needed for cosine normalisation and for Rocchio feedback. Positional information was not stored; this is discussed as a limitation in Section 6."));
kids.push(h2("3.3 Retrieval model selection (Chapters 6, 11 and 12)"));
kids.push(p("Three ranked retrieval models were selected so that the report compares the three major families covered before Chapter 13, and a Boolean AND/OR operator on postings (Chapter 1) is also available through the index for exact-match queries."));
kids.push(tcap("Table 2. Retrieval models implemented"));
kids.push(table(["Model", "Family", "Scoring idea", "Parameters"], [
  ["TF-IDF cosine (lnc.ltc)", "Vector space, Ch. 6", "Log tf with cosine normalisation for documents; log tf × idf for queries", "none"],
  ["Okapi BM25", "Probabilistic, Ch. 11", "idf-weighted, saturating tf with document-length normalisation", "k1, b"],
  ["Query likelihood", "Language model, Ch. 12", "P(q|d) with Jelinek-Mercer smoothing against the collection model", "λ = 0.7"]],
  [1900, 1900, 3626, 1600]));
kids.push(gap());
kids.push(p("BM25 is expected to be strongest on CISI because it combines idf weighting with a tunable length normalisation, whereas the cosine model normalises by the whole vector length. Query likelihood is closely related to the Naive Bayes view in Chapter 13 and serves as a theoretically distinct comparison point."));
kids.push(h2("3.4 Query processing (Chapters 6, 7 and 9)"));
kids.push(p("A query is analysed with the index pipeline, terms absent from the vocabulary are dropped, and the remaining terms are converted to a weighted query vector. Scores are accumulated term-at-a-time: the engine walks the postings list of each query term and adds that term's contribution to a per-document accumulator, so only documents that contain at least one query term are touched. The top k documents are selected with a heap. As an optional second stage, Rocchio pseudo-relevance feedback assumes the top k results are relevant, forms their centroid in tf-idf space and builds an expanded query q' = αq + β·centroid restricted to the m strongest terms, which is then re-scored with the same model (Rocchio, 1971; Manning et al., 2009, Section 9.1.1)."));
kids.push(h2("3.5 Evaluation protocol (Chapters 8 and 13.6)"));
kids.push(p("Effectiveness is measured with precision and recall at 5 and 10, mean average precision (MAP), nDCG@10, mean reciprocal rank and 11-point interpolated precision-recall. Scores are macro-averaged over queries, so each query has equal weight. The 76 judged queries were split into 38 development and 38 test queries (alternating by query ID). BM25 parameters and Rocchio settings were chosen on the development queries only. Final systems are reported on the test half and, for reference, on all 76 queries."));

// ---------------- 4 Implementation
kids.push(h1("4. Implementation"));
kids.push(p("The system is written in Python 3 and organised as a small package. Table 3 lists the modules; the complete code is available in the GitHub repository linked on the title page."));
kids.push(tcap("Table 3. Code organisation"));
kids.push(table(["File", "Responsibility"], [
  ["src/cisi_ir/loader.py", "Parse CISI.ALL, CISI.QRY and CISI.REL (SMART format)"],
  ["src/cisi_ir/preprocess.py", "Tokenisation, stop-word removal, Porter stemming"],
  ["src/cisi_ir/index.py", "Inverted index, statistics, Boolean AND/OR"],
  ["src/cisi_ir/models.py", "TF-IDF, BM25, query likelihood, SearchEngine (top-k)"],
  ["src/cisi_ir/feedback.py", "Rocchio pseudo-relevance feedback"],
  ["src/cisi_ir/evaluate.py", "P@k, R@k, MAP, nDCG, MRR, PR curve"],
  ["run_experiments.py, search_cli.py", "Experiment driver and interactive search interface"]], [3300, 5726]));
kids.push(gap());
kids.push(p("Figure 2 shows the preprocessing component. A single class exposes the three switches (lowercase, stop words, stemming), and a cache avoids re-stemming repeated tokens. Figure 3 shows index construction: each document's term counts are appended to the postings lists in a single pass."));
kids.push(img("results/snapshots/c1_preprocess.png", 470)); kids.push(caption("Figure 2. Preprocessing pipeline (preprocess.py)."));
kids.push(img("results/snapshots/c2_index_build.png", 500)); kids.push(caption("Figure 3. Inverted index construction (index.py)."));
kids.push(p("Figure 4 shows the BM25 scorer. For each query term it iterates over the postings list and adds weight × idf × saturated tf to the document's accumulator, with document length normalised by the collection average. Figure 5 shows the Rocchio expansion step, which keeps only the strongest feedback terms to limit query drift."));
kids.push(img("results/snapshots/c3_bm25.png", 500)); kids.push(caption("Figure 4. BM25 term-at-a-time scoring (models.py)."));
kids.push(img("results/snapshots/c4_rocchio.png", 450)); kids.push(caption("Figure 5. Rocchio pseudo-relevance feedback (feedback.py)."));

// ---------------- 5 Results
kids.push(h1("5. Experiments and Results"));
kids.push(h2("5.1 Effect of preprocessing on the index"));
kids.push(img("results/snapshots/s1_index_stats.png", 540)); kids.push(caption("Figure 6. Console output: index statistics for the three pipelines."));
kids.push(tcap("Table 4. Index statistics by preprocessing pipeline"));
kids.push(table(["Pipeline", "Vocabulary", "Tokens", "Postings"], [
  ["A: tokenise only", "10,013", "187,670", "114,508"],
  ["B: + stop words", "9,889", "104,489", "81,603"],
  ["C: + stop words + stemming", "6,065", "104,489", "76,735"]], [3600, 1800, 1800, 1826]));
kids.push(gap());
kids.push(p("Stop-word removal deletes 44.3% of all tokens while shrinking the vocabulary by only 1.2%, because function words are few but extremely frequent. Stemming then cuts the vocabulary by a further 38.7% (9,889 to 6,065 terms), and postings fall by 33% overall relative to pipeline A. A smaller index means shorter postings lists to traverse at query time."));
kids.push(h2("5.2 Comparison of models and pipelines"));
kids.push(img("results/snapshots/s2_model_comparison.png", 520)); kids.push(caption("Figure 7. Console output: models × preprocessing on all 76 judged queries."));
kids.push(tcap("Table 5. MAP and P@10 on the 76 judged queries"));
kids.push(table(["Pipeline", "TF-IDF MAP / P@10", "BM25 MAP / P@10", "Query LM MAP / P@10"], [
  ["A: tokenise only", "0.089 / 0.149", "0.179 / 0.292", "0.176 / 0.284"],
  ["B: + stop words", "0.137 / 0.230", "0.192 / 0.309", "0.190 / 0.300"],
  ["C: + stop words + stemming", "0.175 / 0.290", "0.222 / 0.367", "0.211 / 0.349"]], [2600, 2142, 2142, 2142]));
kids.push(img("results/fig_map_models.png", 470)); kids.push(caption("Figure 8. MAP for each pipeline and model."));
kids.push(p("Every added preprocessing step improved every model, and the full pipeline C was best throughout. The effect was largest for TF-IDF, whose MAP almost doubled from 0.089 to 0.175, whereas BM25 improved from 0.179 to 0.222. BM25 outperformed the cosine model by 27% in MAP under pipeline C (0.222 versus 0.175) and also led on P@10, nDCG@10 and MRR. Query likelihood with λ = 0.7 was close to BM25 but slightly lower (MAP 0.211). Stemming probably helps because CISI abstracts and the long natural-language queries use different inflections of the same concepts (for example catalogue, catalogues, cataloguing), which exact-term matching would otherwise miss."));
kids.push(h2("5.3 Parameter tuning and pseudo-relevance feedback"));
kids.push(p("A grid search over k1 ∈ {0.6, 0.9, 1.2, 1.5, 1.8, 2.2} and b ∈ {0.25, 0.5, 0.75, 0.9, 1.0} on the 38 development queries selected k1 = 2.2 and b = 0.75 (development MAP 0.237). Rocchio settings were then tuned on the same development queries over feedback documents {3, 5, 10}, feedback terms {10, 20, 40} and β ∈ {0.5, 1.0}, giving 5 documents, 20 terms and β = 1.0."));
kids.push(img("results/fig_bm25_grid.png", 340)); kids.push(caption("Figure 9. BM25 development-set MAP over the (k1, b) grid."));
kids.push(img("results/snapshots/s3_final_systems.png", 590)); kids.push(caption("Figure 10. Console output: parameter selection and final systems."));
kids.push(tcap("Table 6. Final systems (pipeline C)"));
kids.push(table(["System", "MAP (test, 38)", "P@10 (test)", "MAP (all 76)", "P@10 (all 76)"], [
  ["TF-IDF cosine", "0.162", "0.250", "0.175", "0.290"],
  ["BM25 default (k1 = 1.2, b = 0.75)", "0.213", "0.321", "0.222", "0.367"],
  ["BM25 tuned (k1 = 2.2, b = 0.75)", "0.214", "0.300", "0.226", "0.365"],
  ["BM25 tuned + Rocchio PRF", "0.242", "0.337", "0.249", "0.380"]], [3300, 1400, 1400, 1500, 1426]));
kids.push(img("results/fig_pr_curves.png", 400)); kids.push(caption("Figure 11. 11-point interpolated precision-recall curves (76 queries)."));
kids.push(p("Tuning k1 gave essentially no gain on the held-out test queries (MAP 0.214 versus 0.213), and P@10 even dropped from 0.321 to 0.300. This is a useful reminder of the warning in Section 13.6: an improvement on development data does not guarantee an improvement on new data, and with only 38 development queries the differences are within noise. Rocchio pseudo-relevance feedback was the one change that helped on the test half, raising MAP from 0.214 to 0.242 (+13%) and P@10 from 0.300 to 0.337. Over all 76 queries the gain in MAP is smaller (+10%) and R@10 is flat (0.145 to 0.142), which suggests that feedback mainly improves the ordering of the lower part of the ranking rather than the very top. Feedback is not guaranteed to help for every query: when the first-pass top documents are irrelevant, expansion can pull the query away from the information need (query drift)."));
kids.push(h2("5.4 Example query"));
kids.push(img("results/snapshots/s4_sample_query.png", 590)); kids.push(caption("Figure 12. Analysed query and top-10 ranking from the tuned BM25 system (REL = judged relevant)."));
kids.push(p("For Query 5 the analyzer reduces the question to stemmed content terms such as train, inform, retriev and manag. The system places a judged-relevant document on training for on-line retrieval systems at rank 4 and another relevant document at rank 8, while documents about management information systems and information services rank above them because they share many query terms. Figure 13 shows the interactive command-line interface."));
kids.push(img("results/snapshots/s5_cli.png", 440)); kids.push(caption("Figure 13. Command-line search interface."));

// ---------------- 6 Discussion
kids.push(h1("6. Discussion and Limitations"));
kids.push(bullet("The index is not positional, so phrase queries and proximity scoring are unsupported. Adding positions (Chapter 2.4) is the most natural extension."));
kids.push(bullet("The stop list is short and hand-written, and the Porter stemmer is rule-based and sometimes over-stems (for example unobstructed becomes unobstruct). A lemmatiser or a corpus-derived stop list could be compared."));
kids.push(bullet("Relevance judgments are binary and cover 76 of 112 queries. The 38/38 development-test split is small, so differences of about 0.01 MAP should not be treated as significant; I did not run a paired significance test."));
kids.push(bullet("Documents are indexed as a single text field. Weighting the title more heavily (BM25F) might help, since titles are concise topical summaries."));
kids.push(bullet("The reported MAP values (about 0.2 to 0.25) are in the range commonly seen for lexical retrieval on CISI, which is a difficult collection because the queries are long and judged documents often share few exact terms with them. Semantic methods such as dense embeddings would be a sensible future comparison, though they go beyond the methods covered before Chapter 13."));

// ---------------- 7 Conclusion
kids.push(h1("7. Conclusion"));
kids.push(p("I built and evaluated a complete retrieval system on CISI consisting of a configurable preprocessing pipeline, an inverted index, three ranking models, a term-at-a-time query processor and Rocchio feedback. The experiments show that full preprocessing (stop-word removal plus stemming) improved every model, that BM25 was the strongest single model (MAP 0.222 on all judged queries, 27% above TF-IDF cosine), that tuning BM25 parameters on a small development set did not generalise, and that pseudo-relevance feedback gave the most reliable additional gain (MAP 0.242 on test queries). The source code, data-loading scripts and result files are published on GitHub: [paste your GitHub link here]."));

// ---------------- References
kids.push(h1("References"));
const ref = (runs) => new Paragraph({ spacing: { after: 100, line: 276 }, indent: { left: 720, hanging: 720 }, children: runs });
kids.push(ref([t("Dmaso01dsta. (n.d.). "), t("CISI: A dataset for information retrieval", { italics: true }), t(" [Data set]. Kaggle. https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval")]));
kids.push(ref([t("Manning, C. D., Raghavan, P., & Schütze, H. (2009). "), t("An introduction to information retrieval", { italics: true }), t(" (Online ed.). Cambridge University Press. http://www.informationretrieval.org/")]));
kids.push(ref([t("Porter, M. F. (1980). An algorithm for suffix stripping. "), t("Program, 14", { italics: true }), t("(3), 130–137. https://doi.org/10.1108/eb046814")]));
kids.push(ref([t("Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. "), t("Foundations and Trends in Information Retrieval, 3", { italics: true }), t("(4), 333–389. https://doi.org/10.1561/1500000019")]));
kids.push(ref([t("Rocchio, J. J. (1971). Relevance feedback in information retrieval. In G. Salton (Ed.), "), t("The SMART retrieval system: Experiments in automatic document processing", { italics: true }), t(" (pp. 313–323). Prentice-Hall.")]));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } }] },
  numbering: { config: [{ reference: "b", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } }, children: kids }]
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync("/mnt/user-data/outputs/Week6_CISI_IR_Report.docx", b); console.log("written"); });
