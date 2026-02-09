!pip install scispacy 
!pip install benepar
!pip install medspacy
!pip install groq
!pip install requests
!pip install flair

import spacy
import json
import requests
import logging
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
import re
import medspacy
from medspacy.ner import TargetRule
from medspacy.context import ConTextRule
import benepar
from groq import Groq

# CRITICAL FIX: Download benepar model first
import benepar
benepar.download('benepar_en3')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Berkeley Neural Parser
nlp_berkeley = spacy.load("en_core_web_sm")
nlp_berkeley.add_pipe('benepar', config={'model': 'benepar_en3'})
logger.info("Loaded Berkeley Neural Parser for enhanced SRL")

# Load medical-specific spaCy models
try:
    nlp_medical = spacy.load("en_core_sci_sm")
    logger.info("Loaded scientific spaCy model: en_core_sci_sm")
except:
    nlp_medical = spacy.load("en_core_web_sm")
    logger.info("Fallback to standard spaCy model")

nlp_clinical = medspacy.load()
logger.info("Loaded medspaCy clinical model")

nlp = spacy.load("en_core_web_sm")

# Initialize Groq client
groq_client = Groq(api_key='uuuu')

# ENHANCED System prompt for better simple sentence generation
ENHANCED_SYSTEM_PROMPT = """
You are a medical text generation assistant. Your ONLY task is to rewrite medical content into EXTREMELY SIMPLE sentences.

CRITICAL RULES - Follow these EXACTLY:
1. Write ONLY simple sentences with this structure: [SUBJECT] [VERB] [OBJECT]
2. Use ONE sentence per line
3. NO complex sentences, NO compound sentences, NO nested clauses
4. NO pronouns - always use specific nouns
5. NO passive voice - only active voice
6. Each sentence must have exactly ONE action/relationship
7. Break down complex medical procedures into multiple simple steps
8. Use present tense verbs
9. Keep sentences under 10 words when possible

GOOD EXAMPLES:
- "The doctor examines the patient."
- "X-rays show lung damage."
- "Tuberculosis affects the lungs."
- "The test detects bacteria."
- "Patients cough up sputum."

BAD EXAMPLES (DO NOT WRITE LIKE THIS):
- "The medical examination for tuberculosis involves various tests..." (too complex)
- "If the sputum test is positive, then the patient will undergo..." (conditional, complex)
- "This is known as a sputum smear test." (uses pronoun "this")

Your response must be a numbered list of simple sentences ONLY. No explanations, no complex descriptions.
"""

# NEW: System prompt for triple validation
TRIPLE_VALIDATION_PROMPT = """
You are a medical knowledge validation expert. Your task is to evaluate the truthfulness and accuracy of extracted medical triples based on the original text.

EVALUATION CRITERIA:
1. Medical accuracy: Are the relationships medically correct?
2. Factual consistency: Do the triples accurately represent information from the source text?
3. Logical coherence: Do the subject-predicate-object relationships make sense?
4. Completeness: Are important medical relationships captured?

SCORING GUIDELINES:
- 0.9-1.0: Excellent - All triples are medically accurate and well-extracted
- 0.8-0.89: Good - Most triples are accurate with minor issues
- 0.7-0.79: Fair - Some triples are accurate but several have issues
- 0.6-0.69: Poor - Many triples have medical or factual errors
- 0.0-0.59: Very Poor - Most triples are incorrect or nonsensical

Provide ONLY a single decimal number between 0 and 1 as your response. No explanations.
"""

# Configuration for HTTP endpoint
TARGET_ENDPOINT_URL = None  # Will be set by user input
TARGET_HEADERS = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
}

def send_triples_batch_to_endpoint(triples_list, endpoint_url, headers=None):
    """
    Send a list of triples to the specified HTTP endpoint via POST request.
    Args:
        triples_list (list): List of dictionaries with keys 'subject', 'predicate', 'object', 'truth_score'
        endpoint_url (str): The URL to send the POST request to
        headers (dict): Optional HTTP headers
    Returns:
        bool: True if successful, False if error
    """
    try:
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        
        # Prepare the JSON payload with list of triples (each already has truth_score)
        payload = {
            "output": triples_list
        }
        
        # Send POST request
        response = requests.post(
            endpoint_url, 
            json=payload, 
            headers=headers, 
            timeout=10
        )
        response.raise_for_status()
        print(response)
        return True
        
    except Exception as e:
        # Silently handle errors (no printing as per user request)
        return False

def validate_triples_with_groq(triples_list, original_text):
    """
    Validate extracted triples using Groq as fallback validation
    Args:
        triples_list (list): List of extracted triples
        original_text (str): Original LLM generated text
    Returns:
        float: Truth score between 0 and 1
    """
    try:
        # Format triples for validation
        triples_formatted = []
        for i, triple in enumerate(triples_list, 1):
            triples_formatted.append(f"{i}. Subject: {triple['subject']}, Predicate: {triple['predicate']}, Object: {triple['object']}")
        
        triples_text = "\n".join(triples_formatted)
        
        # Create validation prompt
        validation_prompt = f"""
Original Text:
{original_text}

Extracted Triples:
{triples_text}

Evaluate the overall truthfulness and accuracy of these extracted triples based on the original text. Consider medical accuracy, factual consistency, and logical coherence.

Provide a truth score between 0 and 1 (decimal format):"""

        # Call Groq for validation
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": TRIPLE_VALIDATION_PROMPT
                },
                {
                    "role": "user",
                    "content": validation_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.1,  # Low temperature for consistent scoring
            max_tokens=50,
            stop=["\n", "Explanation:", "Reasoning:"]
        )
        
        # Extract and validate score
        score_text = chat_completion.choices[0].message.content.strip()
        print(score_text)
        
        # Parse the score
        try:
            truth_score = float(score_text)
            # Ensure score is between 0 and 1
            truth_score = max(0.0, min(1.0, truth_score))
            logger.info(f"Groq validation score: {truth_score}")
            return truth_score
        except ValueError:
            # If parsing fails, return default score
            logger.warning(f"Failed to parse Groq score: {score_text}, using default 0.5")
            return 0.5
            
    except Exception as e:
        logger.error(f"Groq validation failed: {e}")
        return 0.5  # Default fallback score

def clean_and_deduplicate_text(text):
    """Remove duplicate words while preserving order"""
    if not text:
        return ""
    words = text.split()
    seen = set()
    result = []
    for word in words:
        word_clean = word.lower().strip()
        if word_clean not in seen and word_clean:
            seen.add(word_clean)
            result.append(word)
    return " ".join(result)

def extract_medical_entities_enhanced(text):
    """Enhanced medical entity extraction with position tracking"""
    entities = {}
    entity_variations = {}
    entity_positions = {}
    
    # Process with scientific spaCy model
    doc_sci = nlp_medical(text)
    
    # Process with medspaCy for clinical entities
    doc_clinical = nlp_clinical(text)
    
    # Combine entities from both models
    all_entities = []
    
    # Extract from scientific model
    for ent in doc_sci.ents:
        all_entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'source': 'scispacy'
        })
    
    # Extract from clinical model
    for ent in doc_clinical.ents:
        all_entities.append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'source': 'medspacy'
        })
    
    # Process and categorize medical entities
    for entity_info in all_entities:
        entity_text = clean_and_deduplicate_text(entity_info['text'])
        entity_label = entity_info['label']
        
        # Medical entity categorization
        if entity_label in ['PERSON', 'PER']:
            category = 'MEDICAL_PERSON'
        elif entity_label in ['DISEASE', 'DISORDER', 'PROBLEM', 'CONDITION']:
            category = 'DISEASE'
        elif entity_label in ['DRUG', 'MEDICATION', 'CHEMICAL']:
            category = 'MEDICATION'
        elif entity_label in ['PROCEDURE', 'TREATMENT', 'TEST']:
            category = 'MEDICAL_PROCEDURE'
        elif entity_label in ['ANATOMY', 'BODY_PART', 'ORGAN']:
            category = 'ANATOMY'
        elif entity_label in ['SYMPTOM', 'SIGN']:
            category = 'SYMPTOM'
        elif entity_label in ['ORG', 'ORGANIZATION']:
            category = 'MEDICAL_ORG'
        elif entity_label in ['LOC', 'LOCATION', 'GPE']:
            category = 'LOCATION'
        else:
            category = 'MEDICAL_ENTITY'
        
        # Store canonical entity with position
        if category not in entities or len(entity_text) > len(entities[category]):
            entities[category] = entity_text
            entity_positions[category] = (entity_info['start'], entity_info['end'])
        
        # Store variations
        if category not in entity_variations:
            entity_variations[category] = set()
        entity_variations[category].add(entity_text)
        
        # Add partial variations for multi-word entities
        if ' ' in entity_text:
            words = entity_text.split()
            entity_variations[category].add(words[-1])
            if len(words[0]) > 2:
                entity_variations[category].add(words[0])
    
    return entities, entity_variations, entity_positions

def rule_based_medical_coreference_enhanced(text):
    """Enhanced coreference resolution for medical domain"""
    try:
        entities, entity_variations, entity_positions = extract_medical_entities_enhanced(text)
        
        logger.info(f"Medical entities detected: {entities}")
        
        # Medical role rules
        medical_role_rules = {
            'DOCTOR_ACTIONS': [
                'diagnose', 'diagnosed', 'prescribe', 'prescribed', 'recommend', 'recommended', 
                'refer', 'referred', 'treat', 'treated', 'examine', 'examined', 'operate', 'operated',
                'assess', 'assessed', 'evaluate', 'evaluated', 'consult', 'consulted', 'advise', 'advised',
                'order', 'ordered', 'request', 'requested', 'review', 'reviewed', 'monitor', 'monitored',
                'administer', 'administered', 'inject', 'injected', 'adjust', 'adjusted', 'modify', 'modified',
                'test', 'tested', 'screen', 'screened', 'scan', 'scanned', 'image', 'imaged',
                'perform', 'performed', 'conduct', 'conducted', 'execute', 'executed'
            ],
            
            'PATIENT_ACTIONS': [
                'experience', 'experienced', 'suffer', 'suffered', 'feel', 'felt', 'report', 'reported',
                'develop', 'developed', 'manifest', 'manifested', 'present', 'presented',
                'improve', 'improved', 'recover', 'recovered', 'heal', 'healed', 'respond', 'responded',
                'worsen', 'worsened', 'deteriorate', 'deteriorated', 'decline', 'declined',
                'comply', 'complied', 'adhere', 'adhered', 'follow', 'followed', 'tolerate', 'tolerated'
            ],
            
            'MEDICATION_ACTIONS': [
                'help', 'helped', 'reduce', 'reduced', 'control', 'controlled', 'treat', 'treated',
                'prevent', 'prevented', 'cure', 'cured', 'heal', 'healed', 'relieve', 'relieved',
                'cause', 'caused', 'trigger', 'triggered', 'induce', 'induced'
            ],
            
            'TEST_ACTIONS': [
                'detect', 'detected', 'show', 'showed', 'reveal', 'revealed', 'indicate', 'indicated',
                'confirm', 'confirmed', 'identify', 'identified', 'measure', 'measured'
            ]
        }
        
        # Split and process sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return text
        
        resolved_sentences = [sentences[0]]
        
        # Track medical entities across sentences
        current_doctor = entities.get('MEDICAL_PERSON', 'the doctor')
        current_patient = 'the patient'
        current_medication = entities.get('MEDICATION', 'the medication')
        current_condition = entities.get('DISEASE', 'the condition')
        current_test = entities.get('MEDICAL_PROCEDURE', 'the test')
        
        for sentence in sentences[1:]:
            resolved_sentence = sentence
            
            # Rule-based pronoun resolution based on medical context
            doc = nlp_medical(sentence)
            
            for token in doc:
                if token.text.lower() in ['she', 'he', 'him', 'her']:
                    sentence_verbs = [t.lemma_ for t in doc if t.pos_ == 'VERB']
                    
                    if any(verb in medical_role_rules['DOCTOR_ACTIONS'] for verb in sentence_verbs):
                        resolved_sentence = resolved_sentence.replace(token.text, current_doctor)
                    elif any(verb in medical_role_rules['PATIENT_ACTIONS'] for verb in sentence_verbs):
                        resolved_sentence = resolved_sentence.replace(token.text, current_patient)
                
                elif token.text.lower() in ['it', 'this', 'that', 'they', 'them']:
                    sentence_verbs = [t.lemma_ for t in doc if t.pos_ == 'VERB']
                    
                    if any(verb in medical_role_rules['MEDICATION_ACTIONS'] for verb in sentence_verbs):
                        resolved_sentence = resolved_sentence.replace(token.text, current_medication)
                    elif any(verb in medical_role_rules['TEST_ACTIONS'] for verb in sentence_verbs):
                        resolved_sentence = resolved_sentence.replace(token.text, current_test)
                    else:
                        resolved_sentence = resolved_sentence.replace(token.text, current_condition)
            
            resolved_sentences.append(resolved_sentence)
        
        resolved_text = '. '.join(resolved_sentences) + '.'
        logger.info(f"Enhanced medical coreference resolved")
        return resolved_text
        
    except Exception as e:
        logger.warning(f"Enhanced medical coreference resolution failed: {e}")
        return text

def berkeley_enhanced_srl_comprehensive(sentence, triples):
    """COMPREHENSIVE Berkeley parser SRL for maximum triple extraction"""
    try:
        # Use Berkeley parser for better syntactic analysis
        doc = nlp_berkeley(sentence)
        
        # Extract ALL verbs for comprehensive coverage
        for token in doc:
            if token.pos_ == "VERB":
                predicate = token.lemma_
                
                # COMPREHENSIVE medical verb validation
                medical_verbs = {
                    # Diagnostic verbs
                    'diagnose', 'detect', 'identify', 'discover', 'find', 'locate', 'spot',
                    'examine', 'test', 'screen', 'scan', 'check', 'inspect', 'assess', 'evaluate',
                    
                    # Treatment verbs
                    'treat', 'cure', 'heal', 'repair', 'fix', 'restore', 'recover',
                    'prescribe', 'administer', 'give', 'provide', 'deliver', 'supply',
                    
                    # Patient experience verbs
                    'experience', 'suffer', 'feel', 'have', 'get', 'develop', 'acquire',
                    'show', 'display', 'exhibit', 'manifest', 'present', 'demonstrate',
                    
                    # Medical action verbs
                    'perform', 'conduct', 'carry', 'execute', 'do', 'make', 'create',
                    'operate', 'surgery', 'cut', 'remove', 'insert', 'implant',
                    
                    # Effect verbs
                    'cause', 'trigger', 'induce', 'produce', 'generate', 'create',
                    'prevent', 'stop', 'block', 'inhibit', 'reduce', 'decrease',
                    'increase', 'raise', 'boost', 'enhance', 'improve', 'worsen',
                    
                    # Monitoring verbs
                    'monitor', 'watch', 'observe', 'track', 'follow', 'record',
                    'measure', 'count', 'calculate', 'determine', 'establish',
                    
                    # Communication verbs
                    'recommend', 'suggest', 'advise', 'tell', 'inform', 'explain',
                    'discuss', 'talk', 'speak', 'say', 'report', 'describe',
                    
                    # General medical verbs
                    'affect', 'involve', 'include', 'contain', 'comprise', 'consist',
                    'help', 'assist', 'support', 'aid', 'benefit', 'harm', 'damage'
                }
                
                if predicate not in medical_verbs:
                    continue
                
                # COMPREHENSIVE subject finding
                subject = None
                subject_candidates = []
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                        # Check if subject is part of a medical entity
                        subject_found = False
                        for ent in doc.ents:
                            if child.i >= ent.start and child.i < ent.end:
                                subject_candidates.append(ent.text)
                                subject_found = True
                                break
                        
                        if not subject_found:
                            subj_tokens = []
                            for subchild in child.subtree:
                                if (subchild.dep_ in ["compound", "amod", "det", "poss", "nummod"] or 
                                    subchild == child):
                                    subj_tokens.append(subchild.text)
                            if subj_tokens:
                                subject_candidates.append(" ".join(subj_tokens))
                
                if subject_candidates:
                    subject = max(subject_candidates, key=len)
                
                # COMPREHENSIVE object finding
                obj = None
                obj_candidates = []
                
                for child in token.children:
                    if child.dep_ in ["dobj", "iobj", "attr", "oprd", "ccomp", "xcomp"]:
                        # Check if object is part of a medical entity
                        obj_found = False
                        for ent in doc.ents:
                            if child.i >= ent.start and child.i < ent.end:
                                obj_candidates.append(ent.text)
                                obj_found = True
                                break
                        
                        if not obj_found:
                            obj_tokens = []
                            for subchild in child.subtree:
                                if (subchild.dep_ in ["compound", "amod", "det", "poss", "prep", "pobj", "of", "aux", "nummod"] or 
                                    subchild == child or subchild.pos_ in ["NOUN", "PROPN", "ADJ", "NUM"]):
                                    obj_tokens.append(subchild.text)
                            if obj_tokens:
                                obj_candidates.append(" ".join(obj_tokens))
                    
                    elif child.dep_ == "prep":
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                # Check if prep object is medical entity
                                obj_found = False
                                for ent in doc.ents:
                                    if prep_child.i >= ent.start and prep_child.i < ent.end:
                                        obj_candidates.append(ent.text)
                                        obj_found = True
                                        break
                                
                                if not obj_found:
                                    prep_tokens = []
                                    for subchild in prep_child.subtree:
                                        if (subchild.dep_ in ["compound", "amod", "det", "poss", "nummod"] or 
                                            subchild == prep_child or subchild.pos_ in ["NOUN", "PROPN", "ADJ"]):
                                            prep_tokens.append(subchild.text)
                                    if prep_tokens:
                                        obj_candidates.append(" ".join(prep_tokens))
                
                if obj_candidates:
                    obj = max(obj_candidates, key=len)
                
                # Medical entity fallback - extract ALL possible entities
                if not subject or not obj:
                    entities_in_sentence = [(ent.text, ent.label_) for ent in doc.ents]
                    
                    if not subject:
                        for ent_text, ent_label in entities_in_sentence:
                            if (ent_label in ["PERSON", "ORG", "GPE"] or 
                                any(word in ent_text.lower() for word in ["patient", "doctor", "physician", "test", "medication"])):
                                subject = ent_text
                                break
                    
                    if not obj:
                        for ent_text, ent_label in entities_in_sentence:
                            if ent_text != subject:
                                obj = ent_text
                                break
                
                # Clean and validate
                if subject and obj:
                    subject = clean_and_deduplicate_text(subject)
                    obj = clean_and_deduplicate_text(obj)
                    
                    # Remove possessive forms
                    if subject.endswith("'s"):
                        subject = subject[:-2]
                    
                    # Remove subject from object if it appears
                    if subject.lower() in obj.lower():
                        obj = obj.replace(subject, "").strip()
                        obj = " ".join(obj.split())
                        obj = clean_and_deduplicate_text(obj)
                    
                    if (len(subject) > 0 and len(obj) > 1 and subject.lower() != obj.lower()):
                        triples.append({
                            "subject": subject,
                            "predicate": predicate,
                            "object": obj
                        })
                        logger.info(f"Comprehensive SRL extracted: {subject} -> {predicate} -> {obj}")
                        
    except Exception as e:
        logger.warning(f"Comprehensive SRL extraction failed for sentence '{sentence}': {e}")
    
    return triples

def process_medical_text_comprehensive(text):
    """ENHANCED: Processing with batch triple sending and truth scores added to each dictionary"""
    try:
        # Step 1: Enhanced medical coreference resolution
        resolved_text = rule_based_medical_coreference_enhanced(text)
        logger.info(f"Comprehensive coreference resolved")
        
        # Step 2: Segment sentences using medical model
        doc = nlp_medical(resolved_text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
        logger.info(f"Found {len(sentences)} medical sentences")
        
        # Step 3: Extract triples for every sentence and add to list
        all_triples_list = []  # List of dictionaries for all triples
        
        for i, sent in enumerate(sentences):
            logger.info(f"Processing medical sentence {i+1}: {sent}")
            sentence_triples = []  # Triples for this specific sentence
            sentence_triples = berkeley_enhanced_srl_comprehensive(sent, sentence_triples)
            
            # Add each triple from this sentence to the main list
            for triple in sentence_triples:
                # Clean the triple
                triple['subject'] = clean_and_deduplicate_text(triple['subject'])
                triple['object'] = clean_and_deduplicate_text(triple['object'])
                
                # Add to main list as dictionary
                all_triples_list.append({
                    "subject": triple["subject"],
                    "predicate": triple["predicate"],
                    "object": triple["object"]
                })
        
        # Step 4: Remove duplicates from the list
        unique_triples_list = []
        seen = set()
        for triple_dict in all_triples_list:
            triple_key = (
                triple_dict["subject"].lower().strip(),
                triple_dict["predicate"].lower().strip(),
                triple_dict["object"].lower().strip()
            )
            
            if triple_key not in seen and all(triple_key):
                seen.add(triple_key)
                unique_triples_list.append(triple_dict)
        
        logger.info(f"Total unique medical triples extracted: {len(unique_triples_list)}")
        
        # Step 5: Validate triples with Groq as fallback
        truth_score = 0.5  # Default score
        if unique_triples_list:
            truth_score = validate_triples_with_groq(unique_triples_list, text)
            logger.info(f"Groq validation truth score: {truth_score}")
        
        # Step 6: UPDATED - Add truth_score to every dictionary in the list
        for triple_dict in unique_triples_list:
            triple_dict["truth_score"] = truth_score
        
        # Step 7: Send the entire list of triples to HTTP endpoint
        if TARGET_ENDPOINT_URL and unique_triples_list:
            send_triples_batch_to_endpoint(unique_triples_list, TARGET_ENDPOINT_URL, TARGET_HEADERS)
            logger.info(f"Sent batch of {len(unique_triples_list)} triples to endpoint")
        
        return unique_triples_list, truth_score
        
    except Exception as e:
        logger.error(f"Comprehensive medical text processing failed: {e}")
        return [], 0.5

def generate_simple_medical_text_with_groq(user_prompt):
    """
    Generate SIMPLE medical sentences using Groq API
    """
    try:
        # Call Groq API with ENHANCED system prompt for simple sentences
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": ENHANCED_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"User Request: {user_prompt}\n\nSimple Medical Sentences (numbered list):"
                }
            ],
            model="llama-3.3-70b-versatile",  # Using Groq's fast model
            temperature=0.1,  # Very low temperature for consistent simple structure
            max_tokens=400,
            stop=["User Request:", "BAD EXAMPLES:", "GOOD EXAMPLES:"]
        )
        
        generated_text = chat_completion.choices[0].message.content.strip()
        
        # Clean up the generated text to remove numbering and extra formatting
        lines = generated_text.split('\n')
        clean_sentences = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove numbering (1., 2., etc.)
                line = re.sub(r'^\d+\.\s*', '', line)
                # Remove bullet points
                line = re.sub(r'^[-•*]\s*', '', line)
                if line and len(line) > 5:  # Only keep meaningful sentences
                    clean_sentences.append(line)
        
        final_text = '. '.join(clean_sentences) + '.'
        logger.info("Simple medical sentences generated by Groq LLM")
        return final_text
        
    except Exception as e:
        logger.error(f"Groq simple text generation failed: {e}")
        return f"Error generating text: {str(e)}"

def main():
    """
    Enhanced main function with batch triple sending and truth scores in each dictionary
    """
    global TARGET_ENDPOINT_URL
    
    # Get HTTP endpoint from user
    print("🏥 MEDICAL TRIPLE EXTRACTION SYSTEM WITH BATCH SENDING")
    print("=" * 60)
    print("📡 Enter the HTTP endpoint URL where triples will be sent:")
    endpoint_url = input(">>> ").strip()
    
    if not endpoint_url:
        print("❌ No endpoint provided. Exiting.")
        return
    
    TARGET_ENDPOINT_URL = endpoint_url
    print(f"✅ Endpoint set: {TARGET_ENDPOINT_URL}")
    print("📝 Enter medical prompts. Extracted triples will be sent as a batch with truth scores in each dictionary.")
    print("=" * 60)
    
    while True:
        try:
            # Get user prompt from terminal
            print("\n💬 Enter your medical prompt (or 'quit' to exit):")
            user_prompt = input(">>> ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_prompt:
                continue
            
            # Step 1: Generate SIMPLE medical text using Groq API
            generated_medical_text = generate_simple_medical_text_with_groq(user_prompt)
            print(f"Generated text: {generated_medical_text}")
            
            # Step 2: Process through COMPREHENSIVE medical NLP pipeline with batch sending
            output_list, truth_score = process_medical_text_comprehensive(generated_medical_text)
            
            print(f"Truth Score: {truth_score:.3f}")
            print(f"Extracted {len(output_list)} triples and sent as batch to endpoint")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            # Silently handle errors (no printing as per user request)
            pass

if __name__ == "__main__":
    main()
