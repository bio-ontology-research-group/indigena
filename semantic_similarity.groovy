@Grab(group='com.github.sharispe', module='slib-sml', version='0.9.1')
@Grab(group='net.sourceforge.owlapi', module='owlapi-api', version='4.2.5')
@Grab(group='net.sourceforge.owlapi', module='owlapi-apibinding', version='4.2.5')
@Grab(group='net.sourceforge.owlapi', module='owlapi-impl', version='4.2.5')
@Grab(group='ch.qos.logback', module='logback-classic', version='1.2.3')
@Grab(group='org.slf4j', module='slf4j-api', version='1.7.30')
@Grab(group='org.codehaus.gpars', module='gpars', version='1.1.0')
@Grab('me.tongfei:progressbar:0.9.3')


import org.semanticweb.owlapi.model.*
import  org.semanticweb.owlapi.apibinding.OWLManager

import slib.sml.sm.core.engine.SM_Engine
import slib.sml.sm.core.measures.Measure_Groupwise
import slib.sml.sm.core.metrics.ic.utils.*
import slib.sml.sm.core.utils.SMConstants
import slib.utils.ex.SLIB_Exception
import slib.sml.sm.core.utils.SMconf
import slib.graph.model.impl.graph.memory.GraphMemory
import slib.graph.io.conf.GDataConf
import slib.graph.io.util.GFormat
import slib.graph.io.loader.GraphLoaderGeneric
import slib.graph.model.impl.repo.URIFactoryMemory
import slib.graph.model.impl.graph.elements.*
import slib.sml.sm.core.metrics.ic.utils.*
import slib.graph.algo.utils.*


import org.openrdf.model.vocabulary.RDF

import groovyx.gpars.GParsPool

import java.util.HashSet

import groovy.cli.commons.CliBuilder
import java.nio.file.Paths


import org.slf4j.Logger
import org.slf4j.LoggerFactory

// Initialize the logger

import java.util.logging.Logger
import java.util.logging.Level
import java.util.logging.ConsoleHandler
import java.util.logging.SimpleFormatter

// Initialize the logger
Logger logger = Logger.getLogger(this.class.name)

// Configure the logger
ConsoleHandler handler = new ConsoleHandler()
handler.setLevel(Level.ALL)
handler.setFormatter(new SimpleFormatter())
logger.addHandler(handler)
logger.setLevel(Level.ALL)
logger.setUseParentHandlers(false)




def cli = new CliBuilder(usage: 'semantic_similarity.groovy -r <root_dir> -ic <ic_measure> -pw <pairwise_measure> -gw <groupwise_measure> -fold <fold')
cli.r(longOpt: 'root_dir', args: 1, defaultValue: 'data', 'Root directory')
cli.ic(longOpt: 'ic_measure', args: 1, defaultValue: 'resnik', 'Information Content measure')
cli.pw(longOpt: 'pairwise_measure', args: 1, defaultValue: 'resnik', 'Pairwise measure')
cli.gw(longOpt: 'groupwise_measure', args: 1, defaultValue: 'bma', 'Groupwise measure')
cli.fold(longOpt: 'fold', args: 1, defaultValue: '0', 'Fold number for cross-validation')

def options = cli.parse(args)
if (!options) return

String rootDir = options.r
String icMeasure = options.ic
String pairwiseMeasure = options.pw
String groupwiseMeasure = options.gw

println("Using parameters: icMeasure=${icMeasure}, pairwiseMeasure=${pairwiseMeasure}, groupwiseMeasure=${groupwiseMeasure}, rootDir=${rootDir}, fold=${options.fold}")

def manager = OWLManager.createOWLOntologyManager()
def ontology = manager.loadOntologyFromOntologyDocument(new File(rootDir + "/upheno.owl"))

def classes = ontology.getClassesInSignature().collect { it.toStringID() }

def existingMpPhenotypes = new HashSet()
def existingHpPhenotypes = new HashSet()
classes.each { cls ->
    if (cls.contains("MP_")) {
        existingMpPhenotypes.add(cls)
    } else if (cls.contains("HP_")) {
        existingHpPhenotypes.add(cls)
    }
}

logger.info("Obtaining Gene-Phenotype associations from MGI_GenePheno.rpt. Genes are represented as MGI IDs and Phenotypes are represented as MP IDs")
def gene2pheno = new HashMap()

def mgiGenePhenoFile = new File(rootDir + "/MGI_GenePheno.rpt")
def mgiGenePheno = mgiGenePhenoFile.readLines()*.split('\t')

mgiGenePheno.each { line ->
    def genes = line[6].split("\\|")
    def phenotype = "http://purl.obolibrary.org/obo/" + line[4].replace(":", "_")
    
    if (phenotype in existingMpPhenotypes) {
	genes.each { gene ->
	    gene = "http://mowl.borg/" + gene.replace(":", "_")
	    if (!gene2pheno.containsKey(gene)) {
		gene2pheno[gene] = new HashSet()
	    }
	    gene2pheno[gene].add(phenotype)            
	}
    }
    
}

logger.info("gene2pheno size: ${gene2pheno.size()}")
logger.info("gene2pheno example: ${gene2pheno.take(1)}")

def geneDiseaseFile = new File(rootDir + "/gene_diseases.csv")
def geneDisease = geneDiseaseFile.readLines().tail()*.split(',')
def evalGenes = geneDisease.collect { it[0] }.unique().sort()


testFile =new File(rootDir + "/gene_disease_folds/fold_" + options.fold + "/test.csv")
def testPairs = []

testFile.eachLine { line, lineNumber ->
    // Skip header line
    if (lineNumber > 1) {
	def parts = line.split(',')
	if (parts.length >= 2) {
	    def gene = parts[0]
	    def disease = parts[1]
	    testPairs.add([gene, disease])
	}
    }
}

def testGenes = testPairs.collect { it[0] }.unique().collect { it.split("/").last().replace("_", ":") }
def testDiseases = testPairs.collect { it[1] }.unique().collect { it.split("/").last().replace("_", ":") }

logger.info("Obtaining Disease-Phenotype associations from phenotype.hpoa")
def hpoaFile = new File(rootDir + "/phenotype.hpoa")
def hpoa = hpoaFile.readLines().tail().tail().tail().tail().tail()*.split('\t')

def disease2pheno = new HashMap()

hpoa.each { line ->
    def parts = line
    def disease = "http://mowl.borg/" + parts[0].replace(":", "_")
    def phenotype = "http://purl.obolibrary.org/obo/" + parts[3].replace(":", "_")

    if (existingHpPhenotypes.contains(phenotype)) {
	if (! disease2pheno.containsKey(disease)) {
	    disease2pheno[disease] = []
	}
	disease2pheno[disease].add(phenotype)
    }
}

logger.info("Preparing Semantic Similarity Engine")
def factory = URIFactoryMemory.getSingleton()
def graphUri = factory.getURI("http://purl.obolibrary.org/obo/GDA_")
factory.loadNamespacePrefix("GDA", graphUri.toString())
def graph = new GraphMemory(graphUri)

def goConf = new GDataConf(GFormat.RDF_XML, Paths.get(rootDir, "upheno.owl").toString())
GraphLoaderGeneric.populate(goConf, graph)

def virtualRoot = factory.getURI("http://purl.obolibrary.org/obo/GDA_virtual_root")
def rooting = new GAction(GActionType.REROOTING)
rooting.addParameter("root_uri", virtualRoot.stringValue())
GraphActionExecutor.applyAction(factory, rooting, graph)


def withAnnotations = true

if (withAnnotations) {
    gene2pheno.each { gene, phenotypes ->
	phenotypes.each { phenotype ->
            def geneId = factory.getURI(gene)
	    def phenotypeId = factory.getURI(phenotype)
	    Edge e = new Edge(geneId, RDF.TYPE, phenotypeId)
	    graph.addE(e)
	}
    }
}

def engine = new SM_Engine(graph)

def icConf = null

if (withAnnotations) {
    icConf = new IC_Conf_Corpus(icMeasureResolver(icMeasure))
}
else {
    icConf = new IC_Conf_Topo(icMeasureResolver(icMeasure))
}
    

def smConfPairwise = new SMconf(pairwiseMeasureResolver(pairwiseMeasure))
smConfPairwise.setICconf(icConf)
def smConfGroupwise = new SMconf(groupwiseMeasureResolver(groupwiseMeasure))

def mr = 0
def mrr = 0
def hitsK = [1: 0, 3: 0, 10: 0, 100: 0]
def ranks = [:]

// testPairs = testPairs.collect { [it[0], disease2pheno[it[1]]] }

//get 100 first pairs
// testPairs = testPairs[0..99]

logger.info("Computing Semantic Similarity for ${testPairs.size()} Gene-Disease pairs")
logger.info("Starting Pool ")
 
    def allRanks = GParsPool.withPool {
    testPairs.collectParallel { pair ->

	try 
{
	
	def test_gene = pair[0]
	def test_disease = pair[1]
	def disease_phenotypes = disease2pheno[test_disease].collect { factory.getURI(it) }.toSet()

	if (disease_phenotypes.isEmpty()) {
	    logger.info("No phenotypes found for disease: ${test_disease}")
            
	}
	
	scores = evalGenes.collect { gene ->
            def phenotypes = gene2pheno[gene]
	    def gene_phenotypes = phenotypes.collect { factory.getURI(it) }.toSet()
	    if (gene_phenotypes.isEmpty()) {
		logger.info("No phenotypes found for gene: ${gene}")
            }
	    
	    def sim_score = engine.compare(smConfGroupwise, smConfPairwise, gene_phenotypes, disease_phenotypes)
	    sim_score
	}
	def test_gene_index = evalGenes.indexOf(test_gene)
	
	[test_gene, test_disease, test_gene_index, scores]
	}
	catch (Exception e) {
	    println("Error processing pair ${pair}: ${e.message}")
            
	    // return [pair[0], pair[1], -1, []] // Return empty scores for this pair
	}
    }

}


def out_file = rootDir + "/baseline_results/${icMeasure}_${pairwiseMeasure}_${groupwiseMeasure}_fold${options.fold}_results.txt"
def out = new File(out_file)
out.withWriter { writer ->
	allRanks.each { r ->
	def gene = r[0]
	def disease = r[1]
	def gene_index = r[2]
	def scores = r[3]
	writer.write("${gene}\t${disease}\t${gene_index}\t${scores.join("\t")}\n")
	}
}


logger.info("Done")
// out.close()


logger.info("Results written to ${rootDir}/baseline_results/")



// logger.info("Pool finished. Analyzing results")

static computeRankRoc(ranks, numEntities) {
    def nTails = numEntities

    def aucX = ranks.keySet().sort()
    def aucY = []
    def tpr = 0
    def sumRank = ranks.values().sum()
    aucX.each { x ->
        tpr += ranks[x]
        aucY.add(tpr / sumRank)
    }
    aucX.add(nTails)
    aucY.add(1)
    def auc = 0
        for (int i = 1; i < aucX.size(); i++) {
        auc += (aucX[i] - aucX[i-1]) * (aucY[i] + aucY[i-1]) / 2
    }
    return auc / nTails
}

static icMeasureResolver(measure) {
    if (measure.toLowerCase() == "sanchez") {
        return SMConstants.FLAG_ICI_SANCHEZ_2011
    } else if (measure.toLowerCase() == "resnik") {
	return SMConstants.FLAG_IC_ANNOT_RESNIK_1995_NORMALIZED

    } else {
        throw new IllegalArgumentException("Invalid IC measure: $measure")
    }
}

static pairwiseMeasureResolver(measure) {
    if (measure.toLowerCase() == "lin") {
        return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_LIN_1998
    } else if (measure.toLowerCase() == "resnik") {
	return SMConstants.FLAG_SIM_PAIRWISE_DAG_NODE_RESNIK_1995
    } else {
        throw new IllegalArgumentException("Invalid pairwise measure: $measure")
    }
}

static groupwiseMeasureResolver(measure) {
    if (measure.toLowerCase() == "bma") {
        return SMConstants.FLAG_SIM_GROUPWISE_BMA
    } else if (measure.toLowerCase() == "bmm") {
	return SMConstants.FLAG_SIM_GROUPWISE_BMM
    } else {
        throw new IllegalArgumentException("Invalid groupwise measure: $measure")
    }
}

