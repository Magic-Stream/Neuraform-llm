import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║     ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ███████╗ ██████╗   ║
║     ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██╔════╝██╔═══██╗  ║
║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║█████╗  ██║   ██║  ║
║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██╔══╝  ██║   ██║  ║
║     ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║██║     ╚██████╔╝  ║
║     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝      ╚═════╝   ║
║                                                                    ║
║              ██████╗ ██████╗ ███╗   ███╗                           ║
║              ██╔═══╝██╔═══██╗████╗ ████║                           ║
║              █████╗ ██║   ██║██╔████╔██║                           ║
║              ██╔══╝ ██║   ██║██║╚██╔╝██║                           ║
║              ██║    ╚██████╔╝██║ ╚═╝ ██║                           ║
║              ╚═╝     ╚═════╝ ╚═╝     ╚═╝                           ║
║                                                                    ║
║         🧠 MASSIVE KNOWLEDGE EDITION - Version 2.0 🧠             ║
║              Trained on Extensive Built-in Data                    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {device.upper()}")
if device == 'cuda':
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
#          MASSIVE TRAINING DATA - ALL TOPICS
# ============================================================

KNOWLEDGE_BASE = """

# ===========================================
# SECTION 1: SCIENCE AND PHYSICS
# ===========================================

Physics is the fundamental science that studies matter, energy, space, and time.
The universe is governed by four fundamental forces: gravity, electromagnetism, strong nuclear force, and weak nuclear force.
Albert Einstein developed the theory of relativity, which revolutionized our understanding of space, time, and gravity.
The equation E equals mc squared shows that energy and mass are equivalent and interchangeable.
Isaac Newton discovered the laws of motion and universal gravitation.
Newton's first law states that an object at rest stays at rest unless acted upon by an external force.
Newton's second law states that force equals mass times acceleration.
Newton's third law states that for every action, there is an equal and opposite reaction.
Quantum mechanics describes the behavior of particles at the atomic and subatomic level.
The Heisenberg uncertainty principle states that we cannot simultaneously know both the position and momentum of a particle with perfect precision.
Schrodinger's equation describes how quantum states evolve over time.
The speed of light in a vacuum is approximately 299,792,458 meters per second.
Light exhibits both wave and particle properties, known as wave-particle duality.
Photons are the fundamental particles of light and electromagnetic radiation.
Electrons orbit the nucleus of an atom in discrete energy levels.
Protons and neutrons make up the nucleus of an atom.
Quarks are the fundamental constituents of protons and neutrons.
The Standard Model of particle physics describes all known elementary particles.
The Higgs boson gives other particles their mass through the Higgs field.
Black holes are regions of spacetime where gravity is so strong that nothing can escape.
Stephen Hawking discovered that black holes emit radiation, now called Hawking radiation.
The Big Bang theory explains the origin of the universe approximately 13.8 billion years ago.
Dark matter makes up about 27 percent of the universe but cannot be directly observed.
Dark energy makes up about 68 percent of the universe and causes its accelerated expansion.
Thermodynamics studies heat, energy, and the work done by systems.
The first law of thermodynamics states that energy cannot be created or destroyed, only transformed.
The second law of thermodynamics states that entropy in an isolated system always increases.
Entropy is a measure of disorder or randomness in a system.
Absolute zero is the lowest possible temperature, equal to minus 273.15 degrees Celsius.
Superconductivity is a phenomenon where materials conduct electricity with zero resistance.
Superfluidity is a state of matter that flows without friction.
Plasma is the fourth state of matter, consisting of ionized gas.
Nuclear fusion powers the sun by combining hydrogen atoms into helium.
Nuclear fission splits heavy atoms to release energy, used in nuclear power plants.
Radioactivity is the spontaneous emission of particles or radiation from unstable atomic nuclei.
Half-life is the time required for half of a radioactive substance to decay.
The electromagnetic spectrum includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays.
Sound waves are mechanical vibrations that travel through a medium.
The Doppler effect explains the change in frequency of waves relative to a moving observer.
Resonance occurs when a system oscillates at its natural frequency.
Waves can interfere constructively or destructively when they overlap.
Diffraction is the bending of waves around obstacles or through openings.
Reflection occurs when waves bounce off a surface.
Refraction is the bending of waves as they pass from one medium to another.
Polarization is the orientation of oscillations in a transverse wave.

# ===========================================
# SECTION 2: CHEMISTRY
# ===========================================

Chemistry is the science of matter and the changes it undergoes.
The periodic table organizes all known chemical elements by atomic number.
Atoms are the basic building blocks of all matter.
Molecules are formed when two or more atoms bond together.
Chemical bonds include ionic bonds, covalent bonds, and metallic bonds.
Ionic bonds form when electrons are transferred between atoms.
Covalent bonds form when atoms share electrons.
Hydrogen bonding is a special type of attraction between molecules.
The atomic number equals the number of protons in an atom's nucleus.
The mass number equals the total number of protons and neutrons.
Isotopes are atoms of the same element with different numbers of neutrons.
Electrons occupy orbitals around the nucleus in specific energy levels.
The octet rule states that atoms tend to gain or lose electrons to have eight in their outer shell.
Chemical reactions involve the breaking and forming of chemical bonds.
Reactants are the starting materials in a chemical reaction.
Products are the substances formed in a chemical reaction.
Catalysts speed up chemical reactions without being consumed.
Enzymes are biological catalysts that facilitate reactions in living organisms.
Acids donate hydrogen ions in solution.
Bases accept hydrogen ions in solution.
The pH scale measures the acidity or basicity of a solution.
A pH of 7 is neutral, below 7 is acidic, and above 7 is basic.
Oxidation is the loss of electrons in a chemical reaction.
Reduction is the gain of electrons in a chemical reaction.
Electrochemistry studies the relationship between electricity and chemical reactions.
Batteries convert chemical energy into electrical energy.
Electrolysis uses electrical energy to drive chemical reactions.
Organic chemistry studies compounds containing carbon.
Hydrocarbons contain only hydrogen and carbon atoms.
Functional groups determine the chemical properties of organic molecules.
Polymers are large molecules made of repeating units called monomers.
Plastics are synthetic polymers with diverse applications.
Proteins are polymers of amino acids essential for life.
Carbohydrates include sugars and starches that provide energy.
Lipids include fats and oils that store energy and form cell membranes.
Nucleic acids DNA and RNA store and transmit genetic information.
Biochemistry studies chemical processes in living organisms.
Photosynthesis converts sunlight into chemical energy in plants.
Cellular respiration releases energy from glucose in living cells.
The carbon cycle describes the movement of carbon through Earth's systems.
The nitrogen cycle describes the transformation of nitrogen in the environment.
Water is a polar molecule essential for life.
Solutions are homogeneous mixtures of solutes dissolved in solvents.
Concentration measures the amount of solute in a solution.
Molarity is the number of moles of solute per liter of solution.
Chemical equilibrium occurs when forward and reverse reactions occur at equal rates.
Le Chatelier's principle predicts how systems respond to changes in conditions.
Thermochemistry studies heat changes in chemical reactions.
Exothermic reactions release heat to the surroundings.
Endothermic reactions absorb heat from the surroundings.
Enthalpy is the total heat content of a system.
Gibbs free energy determines whether a reaction is spontaneous.

# ===========================================
# SECTION 3: BIOLOGY AND LIFE SCIENCES
# ===========================================

Biology is the study of living organisms and life processes.
The cell is the basic unit of life.
All living organisms are composed of one or more cells.
Prokaryotic cells lack a nucleus and include bacteria.
Eukaryotic cells have a nucleus and include plants, animals, and fungi.
The cell membrane controls what enters and leaves the cell.
The nucleus contains the cell's genetic material.
DNA is the molecule that carries genetic information.
Genes are segments of DNA that code for proteins.
Chromosomes are structures made of DNA and proteins.
Humans have 23 pairs of chromosomes.
Mitosis is cell division that produces identical daughter cells.
Meiosis is cell division that produces gametes with half the chromosomes.
Genetic mutations are changes in DNA sequences.
Evolution is the change in inherited characteristics over generations.
Charles Darwin proposed the theory of natural selection.
Natural selection favors organisms better adapted to their environment.
Species evolve through random mutations and natural selection.
Fossils provide evidence for evolution and past life forms.
Homologous structures suggest common ancestry among species.
Adaptation is an inherited trait that increases survival and reproduction.
Speciation is the formation of new species.
Biodiversity refers to the variety of life on Earth.
Ecosystems are communities of organisms interacting with their environment.
Food chains show the flow of energy from producers to consumers.
Producers like plants make their own food through photosynthesis.
Consumers obtain energy by eating other organisms.
Decomposers break down dead organic matter.
The water cycle describes the movement of water through Earth's systems.
Climate change affects ecosystems and biodiversity worldwide.
Genetics studies heredity and variation in organisms.
Gregor Mendel discovered the basic laws of inheritance.
Dominant alleles mask the expression of recessive alleles.
Genotype refers to an organism's genetic makeup.
Phenotype refers to an organism's observable characteristics.
Heredity is the passing of traits from parents to offspring.
Genetic engineering modifies organisms by altering their DNA.
Cloning produces genetically identical copies of organisms.
Stem cells can develop into many different cell types.
The human genome contains approximately three billion base pairs.
Proteins are synthesized through transcription and translation.
Transcription copies DNA into messenger RNA.
Translation converts messenger RNA into protein.
Ribosomes are the cellular structures that synthesize proteins.
Amino acids are the building blocks of proteins.
There are twenty standard amino acids used by living organisms.
Protein structure determines protein function.
Enzymes catalyze specific biochemical reactions.
Hormones are chemical messengers that regulate body functions.
The nervous system controls and coordinates body activities.
Neurons transmit electrical signals throughout the body.
The brain is the control center of the nervous system.
The heart pumps blood throughout the circulatory system.
Blood carries oxygen, nutrients, and waste products.
The respiratory system exchanges oxygen and carbon dioxide.
The digestive system breaks down food for absorption.
The immune system defends against pathogens and disease.
Antibodies are proteins that recognize and neutralize foreign invaders.
Vaccines stimulate the immune system to prevent diseases.
Viruses are infectious agents that replicate inside host cells.
Bacteria are single-celled prokaryotic organisms.
Antibiotics kill or inhibit the growth of bacteria.
Antibiotic resistance is a growing public health concern.

# ===========================================
# SECTION 4: MATHEMATICS
# ===========================================

Mathematics is the study of numbers, quantities, and shapes.
Arithmetic deals with basic operations: addition, subtraction, multiplication, and division.
Algebra uses symbols and letters to represent numbers and quantities.
Variables are symbols that represent unknown values.
Equations state that two expressions are equal.
Linear equations have variables raised to the first power.
Quadratic equations have variables raised to the second power.
The quadratic formula solves any quadratic equation.
Polynomials are expressions with multiple terms and variables.
Factoring breaks down expressions into simpler components.
Functions relate inputs to outputs according to specific rules.
The domain of a function is the set of all possible inputs.
The range of a function is the set of all possible outputs.
Linear functions have constant rates of change.
Exponential functions grow or decay at rates proportional to their values.
Logarithms are the inverses of exponential functions.
Trigonometry studies relationships between angles and sides of triangles.
Sine, cosine, and tangent are the primary trigonometric functions.
The Pythagorean theorem relates the sides of a right triangle.
The unit circle defines trigonometric functions for all angles.
Geometry studies shapes, sizes, and properties of space.
Points, lines, and planes are basic geometric concepts.
Angles are formed by two rays sharing a common endpoint.
Triangles have three sides and three angles.
The sum of angles in a triangle equals 180 degrees.
Quadrilaterals have four sides and include squares, rectangles, and parallelograms.
Circles are sets of points equidistant from a center point.
The circumference of a circle equals two times pi times the radius.
The area of a circle equals pi times the radius squared.
Pi is approximately equal to 3.14159.
Volume measures the space inside three-dimensional objects.
Surface area measures the total area of an object's surfaces.
Calculus studies rates of change and accumulation.
Derivatives measure instantaneous rates of change.
The derivative of position with respect to time is velocity.
The derivative of velocity with respect to time is acceleration.
Integrals calculate accumulated quantities over intervals.
The fundamental theorem of calculus connects derivatives and integrals.
Limits describe the behavior of functions as inputs approach specific values.
Infinity represents unbounded quantities.
Statistics analyzes and interprets data.
Mean is the average of a set of numbers.
Median is the middle value in an ordered set.
Mode is the most frequently occurring value.
Standard deviation measures the spread of data around the mean.
Probability measures the likelihood of events occurring.
Probability values range from zero to one.
Independent events do not affect each other's probabilities.
Conditional probability measures likelihood given that another event occurred.
Permutations count arrangements where order matters.
Combinations count selections where order does not matter.
Matrices are rectangular arrays of numbers.
Matrix multiplication combines matrices according to specific rules.
Determinants are scalar values calculated from square matrices.
Vectors have both magnitude and direction.
Vector addition combines vectors geometrically or algebraically.
The dot product of vectors produces a scalar.
The cross product of vectors produces another vector.
Linear algebra studies vector spaces and linear transformations.
Eigenvalues and eigenvectors are fundamental in linear algebra.
Number theory studies properties of integers.
Prime numbers are divisible only by one and themselves.
The fundamental theorem of arithmetic states every integer has a unique prime factorization.
Set theory studies collections of objects.
Union combines elements from multiple sets.
Intersection includes only elements common to all sets.
Mathematical proofs establish truth through logical reasoning.
Induction proves statements for all natural numbers.
Contradiction proves statements by showing the opposite is false.

# ===========================================
# SECTION 5: COMPUTER SCIENCE
# ===========================================

Computer science is the study of computation and information processing.
Algorithms are step-by-step procedures for solving problems.
Data structures organize and store data efficiently.
Arrays store elements in contiguous memory locations.
Linked lists store elements with pointers to next elements.
Stacks follow last-in-first-out ordering.
Queues follow first-in-first-out ordering.
Trees are hierarchical data structures with nodes and edges.
Binary trees have at most two children per node.
Binary search trees keep elements in sorted order.
Hash tables provide fast lookup using hash functions.
Graphs represent relationships between objects.
Sorting algorithms arrange elements in order.
Bubble sort repeatedly swaps adjacent elements.
Quick sort partitions arrays around pivot elements.
Merge sort divides and conquers to sort arrays.
Searching algorithms find elements in data structures.
Binary search efficiently finds elements in sorted arrays.
Time complexity measures how runtime grows with input size.
Space complexity measures memory usage.
Big O notation describes upper bounds on complexity.
Programming languages allow humans to write instructions for computers.
Machine code is the lowest-level programming language.
Assembly language uses mnemonics for machine instructions.
High-level languages like Python are easier for humans to read.
Compilers translate entire programs into machine code.
Interpreters execute programs line by line.
Variables store data values in memory.
Data types define what kinds of values variables can hold.
Integers are whole numbers without decimal points.
Floating-point numbers represent decimal values.
Strings are sequences of characters.
Boolean values are either true or false.
Operators perform operations on values.
Control structures direct the flow of program execution.
Conditional statements execute code based on conditions.
Loops repeat code multiple times.
For loops iterate a specific number of times.
While loops iterate until a condition becomes false.
Functions are reusable blocks of code.
Parameters pass values into functions.
Return values pass results back from functions.
Recursion occurs when functions call themselves.
Object-oriented programming organizes code into objects.
Classes define blueprints for creating objects.
Objects are instances of classes.
Inheritance allows classes to inherit from other classes.
Polymorphism allows objects to take multiple forms.
Encapsulation hides internal details of objects.
Software engineering applies engineering principles to software development.
Requirements analysis determines what software should do.
Design creates the architecture of software systems.
Implementation writes the actual code.
Testing verifies that software works correctly.
Debugging finds and fixes errors in code.
Version control tracks changes to code over time.
Git is a popular version control system.
Databases store and organize large amounts of data.
SQL is a language for querying relational databases.
NoSQL databases handle unstructured data.
Networks connect computers to share data.
The Internet is a global network of networks.
Protocols define rules for communication.
HTTP transfers web pages over the Internet.
TCP ensures reliable data transmission.
IP addresses identify devices on networks.
Encryption protects data from unauthorized access.
Cybersecurity protects systems from attacks.
Artificial intelligence creates systems that can learn and reason.
Machine learning trains computers to learn from data.
Deep learning uses neural networks with many layers.
Neural networks are inspired by biological brains.
Natural language processing enables computers to understand human language.
Computer vision enables computers to interpret images.
Robotics combines hardware and software to create autonomous machines.
Cloud computing provides on-demand computing resources.
Operating systems manage computer hardware and software.
Memory management allocates and deallocates memory.
File systems organize data on storage devices.
Parallel computing performs multiple calculations simultaneously.
Distributed systems spread computation across multiple computers.
Quantum computing uses quantum mechanics for computation.
Qubits can exist in superposition of states.
Quantum entanglement links particles across distances.

# ===========================================
# SECTION 6: HISTORY
# ===========================================

History is the study of past events and human civilization.
Prehistoric times occurred before written records.
The Stone Age was characterized by stone tools.
The Bronze Age saw the development of bronze metallurgy.
The Iron Age introduced iron tools and weapons.
Ancient civilizations developed along major rivers.
Mesopotamia was located between the Tigris and Euphrates rivers.
The Sumerians invented cuneiform writing.
The Babylonians created the Code of Hammurabi.
Ancient Egypt developed along the Nile River.
The Egyptians built the pyramids as tombs for pharaohs.
Hieroglyphics were the writing system of ancient Egypt.
The Indus Valley civilization flourished in South Asia.
Ancient China developed along the Yellow River.
The Chinese invented paper, printing, gunpowder, and the compass.
Ancient Greece contributed to philosophy, democracy, and the arts.
Athens was the birthplace of democracy.
Sparta was known for its military culture.
Socrates, Plato, and Aristotle were influential Greek philosophers.
Alexander the Great created a vast empire.
The Roman Republic established representative government.
The Roman Empire dominated the Mediterranean world.
Julius Caesar was assassinated in 44 BCE.
The fall of Rome occurred in 476 CE.
The Byzantine Empire preserved Roman culture in the east.
The Middle Ages lasted from about 500 to 1500 CE.
Feudalism was the social system of medieval Europe.
The Crusades were religious wars for control of the Holy Land.
The Black Death killed millions of people in Europe.
The Renaissance began in Italy in the 14th century.
Leonardo da Vinci was a Renaissance polymath.
Michelangelo created masterpieces of art and sculpture.
The printing press revolutionized communication.
The Reformation challenged the authority of the Catholic Church.
Martin Luther posted the 95 Theses in 1517.
The Age of Exploration expanded European knowledge of the world.
Christopher Columbus reached the Americas in 1492.
The Scientific Revolution changed how people understood nature.
Galileo Galilei supported the heliocentric model.
The Enlightenment emphasized reason and individual rights.
John Locke influenced ideas about government and liberty.
The American Revolution established the United States.
The Declaration of Independence was signed in 1776.
The French Revolution overthrew the monarchy.
Napoleon Bonaparte conquered much of Europe.
The Industrial Revolution transformed manufacturing and society.
Steam engines powered factories and transportation.
The abolition movement worked to end slavery.
The American Civil War was fought from 1861 to 1865.
World War One lasted from 1914 to 1918.
The Treaty of Versailles ended World War One.
The Great Depression caused worldwide economic hardship.
World War Two lasted from 1939 to 1945.
The Holocaust was the genocide of six million Jews.
The atomic bombs were dropped on Hiroshima and Nagasaki.
The United Nations was founded in 1945.
The Cold War was a rivalry between the USA and USSR.
The Space Race led to the moon landing in 1969.
The Berlin Wall fell in 1989.
The Soviet Union dissolved in 1991.
The Internet transformed global communication.
Globalization increased international connections.
Climate change became a major global concern.

# ===========================================
# SECTION 7: GEOGRAPHY
# ===========================================

Geography is the study of Earth's landscapes, environments, and places.
Earth is the third planet from the Sun.
The Earth has one natural satellite called the Moon.
The Earth's surface is about 71 percent water.
The seven continents are Africa, Antarctica, Asia, Australia, Europe, North America, and South America.
The five oceans are the Pacific, Atlantic, Indian, Southern, and Arctic.
The Pacific Ocean is the largest and deepest ocean.
Mountains are elevated landforms with peaks and slopes.
Mount Everest is the highest mountain above sea level.
The Himalayas are the highest mountain range.
Volcanoes are openings in the Earth's crust where magma escapes.
Earthquakes occur when tectonic plates shift.
The Ring of Fire surrounds the Pacific Ocean.
Plate tectonics explains the movement of Earth's crustal plates.
Continental drift describes how continents have moved over time.
Rivers are flowing bodies of water that drain land.
The Nile is the longest river in the world.
The Amazon River has the largest volume of water.
Lakes are bodies of water surrounded by land.
The Great Lakes are the largest freshwater lakes.
Deserts receive very little precipitation.
The Sahara is the largest hot desert.
Rainforests have high rainfall and biodiversity.
The Amazon rainforest is the largest tropical rainforest.
Glaciers are large masses of ice that move slowly.
Climate refers to long-term weather patterns.
Weather describes short-term atmospheric conditions.
Temperature measures how hot or cold the air is.
Precipitation includes rain, snow, sleet, and hail.
Wind is the movement of air from high to low pressure.
Humidity measures moisture in the air.
The equator divides Earth into Northern and Southern Hemispheres.
Latitude measures distance north or south of the equator.
Longitude measures distance east or west of the Prime Meridian.
Time zones are based on longitude.
Seasons result from Earth's tilted axis.
The Northern Hemisphere has summer when tilted toward the Sun.
Ecosystems are communities of living things and their environment.
Biomes are large ecosystems with similar climate and life.
Tundra is cold and treeless.
Taiga is cold with coniferous forests.
Temperate forests have moderate climates.
Grasslands have few trees and much grass.
Savannas are tropical grasslands with scattered trees.
Coral reefs are underwater ecosystems built by coral.
Wetlands are areas saturated with water.
Population geography studies the distribution of people.
Urbanization is the growth of cities.
Migration is the movement of people from place to place.
Natural resources are materials from nature used by humans.
Renewable resources can be replenished.
Nonrenewable resources exist in limited quantities.
Fossil fuels include coal, oil, and natural gas.
Pollution contaminates the environment.
Conservation protects natural resources and ecosystems.
Sustainable development meets present needs without compromising the future.

# ===========================================
# SECTION 8: LITERATURE AND LANGUAGE
# ===========================================

Literature is written works of artistic merit.
Poetry uses rhythm, imagery, and figurative language.
Prose is ordinary written language without metrical structure.
Fiction tells imaginary stories.
Nonfiction presents factual information.
Novels are long fictional narratives.
Short stories are brief fictional works.
Drama is literature intended for performance.
Tragedy depicts serious themes and often ends in catastrophe.
Comedy is humorous and often ends happily.
Shakespeare was an English playwright and poet.
Hamlet is one of Shakespeare's most famous tragedies.
Romeo and Juliet is a tragedy about star-crossed lovers.
Macbeth explores ambition and guilt.
A Midsummer Night's Dream is a comedy about love and magic.
Homer wrote the Iliad and the Odyssey.
The Iliad tells the story of the Trojan War.
The Odyssey follows Odysseus on his journey home.
Greek mythology includes stories of gods and heroes.
Zeus was the king of the Greek gods.
Athena was the goddess of wisdom.
Roman mythology adapted Greek myths with different names.
Epic poems tell heroic tales on grand scales.
Beowulf is an Old English epic poem.
The Divine Comedy was written by Dante Alighieri.
Paradise Lost was written by John Milton.
Don Quixote is considered the first modern novel.
Pride and Prejudice was written by Jane Austen.
Oliver Twist was written by Charles Dickens.
Moby Dick was written by Herman Melville.
War and Peace was written by Leo Tolstoy.
Crime and Punishment was written by Fyodor Dostoevsky.
The Great Gatsby was written by F. Scott Fitzgerald.
To Kill a Mockingbird was written by Harper Lee.
1984 was written by George Orwell.
Brave New World was written by Aldous Huxley.
The Catcher in the Rye was written by J.D. Salinger.
One Hundred Years of Solitude was written by Gabriel Garcia Marquez.
Metaphors compare unlike things without using like or as.
Similes compare unlike things using like or as.
Personification gives human qualities to nonhuman things.
Alliteration repeats consonant sounds at the beginning of words.
Onomatopoeia uses words that imitate sounds.
Irony expresses meaning opposite to literal words.
Symbolism uses objects to represent ideas.
Foreshadowing hints at future events.
Flashback shows events from the past.
Point of view determines who tells the story.
First person uses I and we.
Third person uses he, she, and they.
Omniscient narrators know everything.
Theme is the central message of a work.
Plot is the sequence of events in a story.
Setting is where and when a story takes place.
Characters are the people or figures in a story.
Conflict is the struggle between opposing forces.
Resolution is how the conflict is resolved.
Language is a system of communication using words and grammar.
Grammar is the set of rules for a language.
Syntax is the arrangement of words in sentences.
Semantics studies the meaning of words and sentences.
Vocabulary is the set of words a person knows.
Etymology studies the origins of words.
Linguistics is the scientific study of language.
Phonetics studies the sounds of speech.
Morphology studies the structure of words.
Sociolinguistics studies language in society.

# ===========================================
# SECTION 9: PHILOSOPHY
# ===========================================

Philosophy is the study of fundamental questions about existence, knowledge, and ethics.
Metaphysics examines the nature of reality.
Epistemology studies the nature of knowledge.
Ethics investigates right and wrong conduct.
Logic studies valid reasoning and arguments.
Aesthetics explores beauty and art.
Political philosophy examines government and justice.
Socrates questioned assumptions to seek truth.
The Socratic method uses questions to stimulate thinking.
Plato wrote dialogues featuring Socrates.
Plato's Republic discusses the ideal state.
The allegory of the cave illustrates the nature of reality.
Aristotle was a student of Plato.
Aristotle wrote on logic, ethics, politics, and science.
Virtue ethics focuses on developing good character.
Stoicism teaches acceptance of what we cannot control.
Epicureanism seeks pleasure through moderation.
Skepticism questions whether knowledge is possible.
Rationalism holds that reason is the source of knowledge.
Empiricism holds that experience is the source of knowledge.
Rene Descartes said I think therefore I am.
Descartes doubted everything to find certain knowledge.
Dualism distinguishes mind from body.
John Locke argued the mind is a blank slate.
David Hume questioned causation and induction.
Immanuel Kant synthesized rationalism and empiricism.
The categorical imperative is Kant's moral principle.
Utilitarianism judges actions by their consequences.
Jeremy Bentham founded utilitarianism.
John Stuart Mill refined utilitarian theory.
The greatest happiness principle seeks the greatest good for the greatest number.
Deontology focuses on duties and rules.
Existentialism emphasizes individual existence and freedom.
Jean-Paul Sartre said existence precedes essence.
Simone de Beauvoir applied existentialism to feminism.
Friedrich Nietzsche criticized traditional morality.
The will to power is a central concept in Nietzsche.
Nihilism denies objective meaning and values.
Phenomenology studies structures of consciousness.
Edmund Husserl founded phenomenology.
Martin Heidegger explored the question of being.
Pragmatism judges ideas by their practical consequences.
William James developed pragmatic philosophy.
John Dewey applied pragmatism to education.
Analytic philosophy emphasizes logical analysis.
Ludwig Wittgenstein studied language and meaning.
The mind-body problem asks how mental and physical relate.
Free will debates whether choices are truly free.
Determinism holds that all events are caused.
Compatibilism reconciles free will with determinism.
The problem of evil questions why suffering exists.
The trolley problem tests moral intuitions.
Social contract theory explains political obligation.
Thomas Hobbes described life without government as nasty brutish and short.
Jean-Jacques Rousseau believed humans are naturally good.
John Rawls proposed the veil of ignorance.
Justice as fairness is Rawls's theory.
Feminist philosophy examines gender and equality.
Environmental ethics considers moral obligations to nature.

# ===========================================
# SECTION 10: PSYCHOLOGY
# ===========================================

Psychology is the scientific study of mind and behavior.
Sigmund Freud founded psychoanalysis.
The unconscious mind contains hidden thoughts and desires.
The id, ego, and superego are parts of personality.
Defense mechanisms protect the ego from anxiety.
Carl Jung developed analytical psychology.
The collective unconscious contains shared human experiences.
Archetypes are universal symbolic patterns.
Behaviorism focuses on observable behavior.
Ivan Pavlov discovered classical conditioning.
Classical conditioning pairs neutral stimuli with responses.
B.F. Skinner developed operant conditioning.
Operant conditioning uses rewards and punishments.
Reinforcement increases the likelihood of behavior.
Punishment decreases the likelihood of behavior.
Cognitive psychology studies mental processes.
Memory stores and retrieves information.
Short-term memory holds information briefly.
Long-term memory stores information indefinitely.
Attention selects information for processing.
Perception interprets sensory information.
Language allows communication through symbols.
Problem solving finds solutions to challenges.
Decision making chooses among alternatives.
Intelligence is the ability to learn and reason.
IQ tests measure cognitive abilities.
Multiple intelligences theory proposes various types of intelligence.
Emotional intelligence involves understanding emotions.
Developmental psychology studies changes across the lifespan.
Jean Piaget described stages of cognitive development.
Sensorimotor stage occurs from birth to two years.
Preoperational stage occurs from two to seven years.
Concrete operational stage occurs from seven to eleven years.
Formal operational stage begins at adolescence.
Erik Erikson proposed stages of psychosocial development.
Identity formation is a key task of adolescence.
Attachment theory describes bonds between caregivers and children.
Secure attachment promotes healthy development.
Social psychology studies how people influence each other.
Conformity is adjusting behavior to match group norms.
Obedience is following commands from authority.
Attitudes are evaluations of people, objects, and ideas.
Persuasion changes attitudes through communication.
Group dynamics affect behavior in social settings.
Prejudice is a negative attitude toward a group.
Stereotypes are generalized beliefs about groups.
Abnormal psychology studies psychological disorders.
Anxiety disorders involve excessive worry and fear.
Depression is characterized by persistent sadness.
Bipolar disorder involves mood swings between mania and depression.
Schizophrenia involves distorted thinking and perception.
Personality disorders are enduring patterns of maladaptive behavior.
Therapy helps people overcome psychological problems.
Psychotherapy uses talk to treat mental issues.
Cognitive behavioral therapy changes thoughts and behaviors.
Humanistic therapy emphasizes personal growth.
Medication can treat symptoms of mental disorders.
Positive psychology studies well-being and happiness.
Resilience is the ability to recover from adversity.
Mindfulness involves present-moment awareness.
Flow is a state of complete absorption in activity.

# ===========================================
# SECTION 11: ECONOMICS
# ===========================================

Economics studies how societies allocate scarce resources.
Microeconomics examines individual and business decisions.
Macroeconomics studies the economy as a whole.
Supply is the quantity of goods producers offer.
Demand is the quantity of goods consumers want.
The law of supply states that higher prices increase supply.
The law of demand states that higher prices decrease demand.
Equilibrium occurs where supply equals demand.
Markets are where buyers and sellers exchange goods.
Competition among sellers benefits consumers.
Monopolies exist when one firm dominates a market.
Oligopolies have few competing firms.
Elasticity measures responsiveness to price changes.
Gross domestic product measures total economic output.
GDP growth indicates economic expansion.
Inflation is a general increase in prices.
Deflation is a general decrease in prices.
Unemployment occurs when people cannot find jobs.
The labor force includes all people working or seeking work.
Fiscal policy uses government spending and taxation.
Monetary policy controls the money supply and interest rates.
Central banks regulate monetary policy.
The Federal Reserve is the central bank of the United States.
Interest rates affect borrowing and spending.
Banks accept deposits and make loans.
Credit allows people to borrow money.
Debt is money owed to lenders.
Investment allocates resources for future returns.
Stocks represent ownership in companies.
Bonds are loans to governments or corporations.
Diversification spreads risk across investments.
Trade allows countries to specialize and exchange goods.
Exports are goods sold to other countries.
Imports are goods bought from other countries.
Tariffs are taxes on imported goods.
Free trade removes barriers between countries.
Globalization increases international economic integration.
Exchange rates determine the value of currencies.
Balance of trade compares exports and imports.
Economic development improves living standards.
Poverty is the lack of sufficient resources.
Inequality refers to uneven distribution of wealth.
Public goods benefit everyone and are nonexcludable.
Externalities are costs or benefits affecting third parties.
Market failure occurs when markets produce inefficient outcomes.
Government intervention can correct market failures.
Regulation sets rules for business behavior.
Taxation funds government activities.
Progressive taxes take more from higher incomes.
Regressive taxes take more from lower incomes.
Budgets plan government revenues and expenditures.
Deficits occur when spending exceeds revenue.
Surpluses occur when revenue exceeds spending.
National debt is the total amount owed by government.
Capitalism is an economic system based on private ownership.
Socialism is an economic system based on collective ownership.
Mixed economies combine elements of capitalism and socialism.

# ===========================================
# SECTION 12: ART AND MUSIC
# ===========================================

Art is the expression of human creativity and imagination.
Visual arts include painting, sculpture, and photography.
Painting applies pigment to surfaces.
Oil painting uses pigments mixed with oil.
Watercolor uses water-soluble pigments.
Acrylic paint is fast-drying and versatile.
Sculpture creates three-dimensional forms.
Architecture designs buildings and structures.
Photography captures images using light.
Drawing uses lines to create images.
Printmaking transfers images from surfaces to paper.
Ceramics creates objects from clay.
The Renaissance produced masterpieces of art.
Leonardo da Vinci painted the Mona Lisa.
Michelangelo painted the Sistine Chapel ceiling.
Baroque art is dramatic and ornate.
Rembrandt was a Dutch Baroque master.
Impressionism captured light and movement.
Claude Monet painted water lilies.
Post-Impressionism developed new styles.
Vincent van Gogh painted Starry Night.
Expressionism conveyed emotional experience.
Cubism fragmented forms into geometric shapes.
Pablo Picasso co-founded Cubism.
Surrealism explored dreams and the unconscious.
Salvador Dali painted melting clocks.
Abstract art does not represent reality.
Pop art drew from popular culture.
Andy Warhol created iconic pop art images.
Contemporary art includes diverse current practices.
Music is the art of organized sound.
Melody is a sequence of musical notes.
Harmony combines multiple notes simultaneously.
Rhythm is the pattern of beats in music.
Tempo is the speed of music.
Dynamics refer to loudness and softness.
Pitch is how high or low a sound is.
Timbre is the quality of a sound.
Scales are sequences of notes in order.
Major scales sound happy and bright.
Minor scales sound sad and dark.
Chords are groups of notes played together.
Orchestras are large instrumental ensembles.
Symphonies are extended orchestral compositions.
Concertos feature solo instruments with orchestra.
Chamber music is for small groups.
Opera combines music, drama, and staging.
Classical music includes works from the Classical period.
Johann Sebastian Bach was a Baroque composer.
Wolfgang Amadeus Mozart was a Classical genius.
Ludwig van Beethoven bridged Classical and Romantic eras.
Romantic music emphasized emotion and individualism.
Jazz originated in African American communities.
Blues influenced the development of jazz and rock.
Rock and roll emerged in the 1950s.
The Beatles revolutionized popular music.
Hip hop originated in the Bronx in the 1970s.
Electronic music uses electronic instruments and technology.
Folk music reflects cultural traditions.
Country music originated in the American South.
Reggae developed in Jamaica.
Musical instruments produce sound.
String instruments include violin, guitar, and piano.
Wind instruments include flute, clarinet, and trumpet.
Percussion instruments include drums and cymbals.
Keyboards include piano and organ.

# ===========================================
# SECTION 13: TECHNOLOGY AND ENGINEERING
# ===========================================

Technology applies scientific knowledge for practical purposes.
Engineering designs and builds machines, structures, and systems.
Mechanical engineering deals with machines and mechanisms.
Electrical engineering works with electricity and electronics.
Civil engineering designs infrastructure like roads and bridges.
Chemical engineering applies chemistry to industrial processes.
Aerospace engineering designs aircraft and spacecraft.
Biomedical engineering applies engineering to medicine.
Computer engineering designs computer hardware.
Software engineering develops computer programs.
Materials engineering develops new materials.
Environmental engineering protects the environment.
Industrial engineering optimizes complex systems.
Machines convert energy into useful work.
Simple machines include levers, pulleys, and inclined planes.
Levers multiply force using a fulcrum.
Pulleys change the direction of force.
Inclined planes reduce the force needed to raise objects.
Wheels reduce friction in transportation.
Gears transmit rotational motion.
Engines convert fuel into mechanical energy.
Internal combustion engines burn fuel inside cylinders.
Electric motors convert electrical energy into motion.
Generators convert mechanical energy into electricity.
Turbines use fluid flow to produce rotation.
Transformers change voltage levels in electrical systems.
Transistors are basic components of electronic circuits.
Integrated circuits contain many transistors on chips.
Microprocessors are the brains of computers.
Sensors detect changes in the environment.
Actuators produce physical motion.
Robots are machines that can perform tasks autonomously.
Automation reduces the need for human intervention.
Manufacturing produces goods from raw materials.
Assembly lines organize production into sequential steps.
Quality control ensures products meet standards.
3D printing creates objects layer by layer.
Nanotechnology works at the molecular scale.
Biotechnology applies biology to develop products.
Genetic engineering modifies DNA for specific purposes.
Renewable energy comes from sustainable sources.
Solar panels convert sunlight into electricity.
Wind turbines harness wind energy.
Hydroelectric power uses flowing water.
Geothermal energy uses heat from the Earth.
Nuclear power generates electricity from atomic reactions.
Batteries store electrical energy chemically.
Fuel cells convert chemical energy directly to electricity.
Smart grids optimize electricity distribution.
Electric vehicles use electric motors instead of combustion engines.
Autonomous vehicles can drive without human control.
The Internet of Things connects everyday devices.
Artificial intelligence enables machines to learn and reason.
Virtual reality creates immersive digital environments.
Augmented reality overlays digital information on the real world.
Blockchain is a distributed ledger technology.
Cryptography protects information using codes.
Cybersecurity defends against digital threats.

# ===========================================
# SECTION 14: MEDICINE AND HEALTH
# ===========================================

Medicine is the science of diagnosing, treating, and preventing disease.
Anatomy studies the structure of the body.
Physiology studies how the body functions.
Pathology studies the causes and effects of disease.
Pharmacology studies drugs and their effects.
Surgery treats conditions through operations.
Internal medicine treats diseases of internal organs.
Pediatrics focuses on children's health.
Geriatrics focuses on elderly health.
Psychiatry treats mental disorders.
Cardiology treats heart conditions.
Neurology treats nervous system disorders.
Oncology treats cancer.
Dermatology treats skin conditions.
Orthopedics treats musculoskeletal conditions.
Ophthalmology treats eye conditions.
Radiology uses imaging to diagnose disease.
Pathology examines tissues for disease.
Emergency medicine treats acute conditions.
Preventive medicine focuses on disease prevention.
Diagnosis identifies the nature of illness.
Symptoms are signs of disease experienced by patients.
Physical examination assesses the body.
Laboratory tests analyze blood and other samples.
Medical imaging includes X-rays, CT scans, and MRI.
X-rays use radiation to create images of bones.
CT scans create detailed cross-sectional images.
MRI uses magnetic fields to image soft tissues.
Ultrasound uses sound waves for imaging.
Treatment aims to cure or manage disease.
Medications are drugs used to treat conditions.
Antibiotics treat bacterial infections.
Antivirals treat viral infections.
Vaccines prevent infectious diseases.
Immunization stimulates the immune system.
Chemotherapy treats cancer with drugs.
Radiation therapy uses radiation to kill cancer cells.
Transplants replace failed organs.
Prosthetics replace missing body parts.
Physical therapy helps restore movement and function.
Nutrition affects health and disease risk.
A balanced diet includes all essential nutrients.
Vitamins are essential organic compounds.
Minerals are essential inorganic elements.
Proteins build and repair body tissues.
Carbohydrates provide energy.
Fats store energy and support cell function.
Fiber aids digestion.
Hydration maintains body fluid balance.
Exercise benefits physical and mental health.
Cardiovascular exercise strengthens the heart.
Strength training builds muscle.
Flexibility exercises improve range of motion.
Sleep is essential for health and recovery.
Stress affects physical and mental well-being.
Mental health is as important as physical health.
Public health protects community health.
Epidemiology studies disease patterns in populations.
Sanitation prevents the spread of disease.
Clean water is essential for health.
Vaccination programs control infectious diseases.
Health education promotes healthy behaviors.
Healthcare systems provide medical services.
Primary care is the first point of contact.
Specialists treat specific conditions.
Hospitals provide intensive care.
Clinics offer outpatient services.
Health insurance helps pay for medical care.
Medical research advances understanding and treatment.
Clinical trials test new treatments.
Evidence-based medicine uses research to guide practice.

# ===========================================
# SECTION 15: ENVIRONMENT AND ECOLOGY
# ===========================================

Ecology studies the relationships between organisms and their environment.
Ecosystems are communities of living things interacting with their surroundings.
Biomes are large ecological areas with distinct climates and life.
Habitats are places where organisms live.
Niches are the roles organisms play in ecosystems.
Food webs show complex feeding relationships.
Producers make their own food through photosynthesis.
Primary consumers eat producers.
Secondary consumers eat primary consumers.
Tertiary consumers eat secondary consumers.
Decomposers break down dead organic matter.
Energy flows through ecosystems from producers to consumers.
Only about 10 percent of energy transfers between trophic levels.
Nutrients cycle through ecosystems.
The carbon cycle moves carbon through the environment.
The nitrogen cycle transforms nitrogen into usable forms.
The water cycle moves water through evaporation and precipitation.
The phosphorus cycle moves phosphorus through rocks and organisms.
Biodiversity is the variety of life in an ecosystem.
Species diversity measures the number of different species.
Genetic diversity measures variation within species.
Ecosystem diversity measures the variety of ecosystems.
Biodiversity supports ecosystem stability and resilience.
Keystone species have disproportionate effects on ecosystems.
Invasive species disrupt native ecosystems.
Extinction is the permanent loss of species.
Mass extinctions have occurred throughout Earth's history.
The current extinction rate is higher than normal.
Conservation protects species and ecosystems.
Protected areas preserve natural habitats.
National parks conserve natural and cultural resources.
Wildlife reserves protect animal populations.
Endangered species are at risk of extinction.
The Endangered Species Act protects threatened species.
Habitat loss is the leading cause of species decline.
Deforestation destroys forest ecosystems.
Wetlands filter water and provide habitat.
Coral reefs support immense biodiversity.
Ocean acidification threatens marine life.
Pollution degrades environmental quality.
Air pollution includes harmful gases and particles.
Water pollution contaminates aquatic ecosystems.
Soil pollution affects land and food production.
Plastic pollution threatens wildlife and ecosystems.
Climate change alters global weather patterns.
Greenhouse gases trap heat in the atmosphere.
Carbon dioxide is the main greenhouse gas from human activities.
Methane is a potent greenhouse gas.
Global warming increases average temperatures.
Sea level rise threatens coastal areas.
Extreme weather events are becoming more frequent.
Mitigation reduces greenhouse gas emissions.
Adaptation adjusts to climate change impacts.
Renewable energy reduces carbon emissions.
Energy efficiency decreases energy consumption.
Sustainable practices meet current needs without compromising the future.
Recycling reprocesses waste into new products.
Composting converts organic waste into fertilizer.
Reducing consumption decreases environmental impact.
Sustainable agriculture protects natural resources.
Organic farming avoids synthetic chemicals.
Permaculture designs sustainable food systems.
Environmental policy addresses environmental problems.
International agreements coordinate global action.
The Paris Agreement aims to limit global warming.
Environmental education raises awareness.
Individual actions contribute to environmental protection.

# ===========================================
# SECTION 16: SPACE AND ASTRONOMY
# ===========================================

Astronomy is the study of celestial objects and the universe.
The universe contains all matter, energy, space, and time.
The Big Bang theory explains the origin of the universe.
The universe is approximately 13.8 billion years old.
The universe continues to expand.
Galaxies are large systems of stars, gas, and dust.
The Milky Way is our home galaxy.
The Milky Way contains hundreds of billions of stars.
Spiral galaxies have arms extending from a central bulge.
Elliptical galaxies are rounded and lack spiral arms.
Irregular galaxies have no distinct shape.
Galaxy clusters contain hundreds to thousands of galaxies.
Dark matter is invisible matter detected by gravitational effects.
Dark energy causes the accelerated expansion of the universe.
Stars are luminous spheres of plasma.
Stars form from collapsing clouds of gas and dust.
Nuclear fusion powers stars by combining hydrogen into helium.
The sun is a medium-sized star.
The sun is about 4.6 billion years old.
The sun's core reaches temperatures of 15 million degrees Celsius.
Solar flares are eruptions of energy from the sun.
Sunspots are cooler regions on the sun's surface.
The solar wind is a stream of charged particles from the sun.
Red giants are large evolved stars.
White dwarfs are dense remnants of medium-sized stars.
Supernovae are explosive deaths of massive stars.
Neutron stars are extremely dense stellar remnants.
Pulsars are rotating neutron stars emitting radiation.
Black holes are regions where gravity prevents light from escaping.
Supermassive black holes exist at the centers of galaxies.
The solar system formed about 4.6 billion years ago.
The sun contains 99.8 percent of the solar system's mass.
Mercury is the closest planet to the sun.
Venus is the hottest planet due to its greenhouse effect.
Earth is the only known planet with life.
Mars is called the Red Planet.
The asteroid belt lies between Mars and Jupiter.
Jupiter is the largest planet in the solar system.
Jupiter's Great Red Spot is a giant storm.
Saturn is known for its prominent rings.
Uranus rotates on its side.
Neptune has the strongest winds in the solar system.
Pluto is classified as a dwarf planet.
Moons are natural satellites orbiting planets.
Earth's moon influences tides.
Europa may have a subsurface ocean.
Titan has a thick atmosphere and liquid methane.
Comets are icy bodies that develop tails near the sun.
Asteroids are rocky objects orbiting the sun.
Meteoroids are small particles in space.
Meteors are meteoroids that burn up in Earth's atmosphere.
Meteorites are meteoroids that reach Earth's surface.
Exoplanets are planets orbiting other stars.
Thousands of exoplanets have been discovered.
The habitable zone is the region where liquid water can exist.
The search for extraterrestrial life continues.
Space telescopes observe the universe from orbit.
The Hubble Space Telescope has made groundbreaking discoveries.
The James Webb Space Telescope observes in infrared.
Space exploration has sent probes throughout the solar system.
The Apollo missions landed humans on the moon.
The International Space Station orbits Earth.
Mars rovers explore the Martian surface.
SpaceX and other companies are developing reusable rockets.
Plans exist for human missions to Mars.
Space colonization may extend human civilization beyond Earth.

# ===========================================
# SECTION 17: RELIGION AND MYTHOLOGY
# ===========================================

Religion is a system of beliefs and practices concerning the sacred.
Religions often involve worship of deities or spiritual beings.
Monotheism is belief in one god.
Polytheism is belief in multiple gods.
Atheism is the absence of belief in gods.
Agnosticism holds that the existence of gods is unknown.
Christianity is based on the teachings of Jesus Christ.
Christians believe Jesus is the Son of God.
The Bible is the holy scripture of Christianity.
The Old Testament contains Jewish scriptures.
The New Testament contains the Gospels and other writings.
Catholics recognize the Pope as the leader of the Church.
Protestants broke from the Catholic Church in the Reformation.
Orthodox Christianity is prominent in Eastern Europe.
Islam is based on the teachings of the Prophet Muhammad.
Muslims believe in one God called Allah.
The Quran is the holy book of Islam.
The Five Pillars of Islam guide Muslim practice.
Mecca is the holiest city in Islam.
Sunni and Shia are the two main branches of Islam.
Judaism is one of the oldest monotheistic religions.
Jews follow the Torah as their sacred text.
The Talmud contains Jewish teachings and law.
Jerusalem is a holy city for Jews, Christians, and Muslims.
Hinduism is a major religion of South Asia.
Hindus believe in many gods as manifestations of one reality.
Brahma, Vishnu, and Shiva are important Hindu deities.
The Vedas are ancient Hindu scriptures.
Karma is the law of cause and effect.
Reincarnation is the rebirth of the soul in new bodies.
Buddhism was founded by Siddhartha Gautama, the Buddha.
Buddhists seek enlightenment through the Eightfold Path.
The Four Noble Truths explain suffering and its cessation.
Nirvana is the ultimate goal of Buddhist practice.
Zen Buddhism emphasizes meditation.
Sikhism was founded by Guru Nanak.
Sikhs believe in one God and equality of all people.
The Guru Granth Sahib is the Sikh holy scripture.
Confucianism emphasizes ethics and social harmony.
Taoism seeks harmony with the Tao or Way.
Shinto is the indigenous religion of Japan.
Mythology includes traditional stories of gods and heroes.
Greek mythology features gods like Zeus, Athena, and Apollo.
Roman mythology adapted Greek myths with different names.
Norse mythology features gods like Odin, Thor, and Loki.
Egyptian mythology features gods like Ra, Osiris, and Isis.
Hindu mythology includes epics like the Mahabharata and Ramayana.
Chinese mythology includes stories of dragons and immortals.
Native American mythology varies among tribes.
African mythology reflects diverse cultures.
Myths often explain natural phenomena and human nature.
Creation myths describe the origin of the world.
Flood myths appear in many cultures.
Hero myths follow patterns of adventure and transformation.
Rituals are ceremonial acts with symbolic meaning.
Festivals celebrate religious events and seasons.
Prayer is communication with the divine.
Meditation focuses the mind for spiritual purposes.
Pilgrimage is travel to sacred places.
Religious art expresses spiritual themes.
Sacred music enhances worship and devotion.

# ===========================================
# SECTION 18: SPORTS AND GAMES
# ===========================================

Sports are physical activities involving competition and skill.
Athletics includes track and field events.
Running events range from sprints to marathons.
Jumping events include long jump and high jump.
Throwing events include shot put, discus, and javelin.
Swimming is racing through water using various strokes.
Gymnastics involves acrobatic exercises on apparatus.
Team sports require cooperation among players.
Football is the most popular sport worldwide.
American football uses an oval ball and allows tackling.
Basketball was invented by James Naismith.
Baseball is called America's pastime.
Cricket is popular in Commonwealth countries.
Rugby is a physical contact sport.
Hockey is played on ice or field.
Volleyball is played over a net.
Tennis is played with rackets on a court.
Golf involves hitting balls into holes on a course.
Martial arts include judo, karate, taekwondo, and others.
Boxing involves fighting with fists.
Wrestling involves grappling with opponents.
Cycling includes road racing and mountain biking.
Motor sports include car and motorcycle racing.
Formula One is the premier class of auto racing.
Winter sports include skiing, snowboarding, and skating.
Alpine skiing involves racing downhill on snow.
Cross-country skiing covers long distances on flat terrain.
Figure skating combines athleticism with artistic expression.
Speed skating races around an oval ice track.
Water sports include surfing, sailing, and water polo.
Extreme sports involve high levels of risk.
The Olympic Games are the largest international sporting event.
The ancient Olympics were held in Greece.
The modern Olympics began in 1896.
The Summer Olympics feature warm-weather sports.
The Winter Olympics feature cold-weather sports.
The Paralympic Games feature athletes with disabilities.
Professional athletes compete for pay.
Amateur athletes compete without payment.
Coaches train and guide athletes.
Referees enforce rules during competition.
Sportsmanship involves fair and ethical behavior.
Doping is the use of prohibited substances.
Sports medicine treats athletic injuries.
Physical fitness improves athletic performance.
Training develops skills and conditioning.
Strategy and tactics influence game outcomes.
Records mark the best performances achieved.
Championships determine the best teams or individuals.
Fans support their favorite teams and athletes.
Sports culture influences society and identity.
Video games simulate sports and other activities.
Esports are competitive video game competitions.
Board games include chess, checkers, and Monopoly.
Chess is a strategic game with ancient origins.
Card games include poker, bridge, and solitaire.
Puzzle games challenge problem-solving skills.
Role-playing games involve assuming fictional characters.
Strategy games require planning and decision-making.
Games provide entertainment and social interaction.

# ===========================================
# SECTION 19: FOOD AND COOKING
# ===========================================

Cooking transforms raw ingredients into prepared food.
Culinary arts are the skills of preparing and presenting food.
Recipes provide instructions for making dishes.
Ingredients are the components used in cooking.
Measuring ensures accurate proportions.
Cutting techniques prepare ingredients for cooking.
Dicing cuts food into small cubes.
Mincing cuts food into very small pieces.
Julienne cuts food into thin strips.
Heat transforms food through cooking.
Boiling cooks food in water at 100 degrees Celsius.
Simmering cooks food in liquid just below boiling.
Steaming cooks food using steam from boiling water.
Poaching cooks food gently in liquid.
Blanching briefly cooks food in boiling water.
Frying cooks food in hot oil or fat.
Sautéing cooks food quickly in a small amount of fat.
Stir-frying cooks food rapidly over high heat.
Deep-frying cooks food submerged in hot oil.
Roasting cooks food in an oven with dry heat.
Baking cooks food in an oven, especially breads and pastries.
Grilling cooks food over direct heat.
Broiling cooks food under direct heat.
Braising cooks food slowly in liquid after browning.
Smoking preserves and flavors food with smoke.
Seasoning adds flavor to food.
Salt enhances flavor and preserves food.
Pepper adds spiciness to dishes.
Herbs are leaves used for flavoring.
Spices are seeds, bark, roots, or fruits used for flavoring.
Garlic adds pungent flavor to dishes.
Onions provide a base for many recipes.
Sauces add flavor and moisture to dishes.
Stocks are flavorful liquids made by simmering ingredients.
Emulsions combine ingredients that normally do not mix.
Marinades soak food to add flavor and tenderize.
Fermentation uses microorganisms to transform food.
Bread is made from flour, water, yeast, and salt.
Yeast causes bread to rise by producing carbon dioxide.
Pasta is made from wheat flour and water or eggs.
Rice is a staple grain for billions of people.
Vegetables provide vitamins, minerals, and fiber.
Fruits provide natural sweetness and nutrients.
Meat includes beef, pork, poultry, and lamb.
Seafood includes fish, shellfish, and crustaceans.
Dairy products include milk, cheese, and yogurt.
Eggs are versatile ingredients used in many dishes.
Nuts and seeds provide protein and healthy fats.
Legumes include beans, lentils, and peas.
Tofu is made from soybeans and is high in protein.
Olive oil is a healthy fat used in Mediterranean cooking.
Butter adds richness to dishes.
Sugar adds sweetness and helps with browning.
Chocolate is made from cacao beans.
Coffee is a popular caffeinated beverage.
Tea is made from the leaves of the Camellia sinensis plant.
Wine is an alcoholic beverage made from fermented grapes.
Beer is made from fermented grains.
Cuisines reflects the food traditions of different cultures.
Italian cuisine features pasta, pizza, and olive oil.
French cuisine is known for its techniques and sauces.
Chinese cuisine varies by region and includes stir-frying.
Japanese cuisine features sushi, ramen, and delicate flavors.
Indian cuisine uses complex spice combinations.
Mexican cuisine features corn, beans, and chili peppers.
Mediterranean cuisine emphasizes fresh vegetables and olive oil.
Thai cuisine balances sweet, sour, salty, and spicy flavors.
Vegetarian diets exclude meat.
Vegan diets exclude all animal products.
Food safety prevents illness from contaminated food.
Nutrition science studies how food affects health.
Sustainable food systems protect the environment.

# ===========================================
# SECTION 20: GENERAL KNOWLEDGE
# ===========================================

Knowledge is information acquired through experience or education.
Facts are statements that can be verified as true.
Opinions are personal beliefs that may not be verifiable.
Critical thinking evaluates information objectively.
Logic is the study of valid reasoning.
Evidence supports claims with data and facts.
The scientific method tests hypotheses through experiments.
Observation gathers information about the world.
Hypotheses are proposed explanations to be tested.
Experiments test hypotheses under controlled conditions.
Data are collected observations and measurements.
Analysis interprets data to draw conclusions.
Peer review evaluates research by experts.
Theories are well-supported explanations of phenomena.
Laws describe consistent patterns in nature.
Education develops knowledge and skills.
Literacy is the ability to read and write.
Numeracy is the ability to work with numbers.
Schools provide formal education.
Universities offer higher education and research.
Libraries store and provide access to information.
Museums preserve and display cultural artifacts.
The Internet provides access to vast information.
Search engines help find information online.
Social media enables sharing of content.
News media report on current events.
Journalism investigates and reports information.
Freedom of the press protects journalists.
Democracy is government by the people.
Citizens participate in democratic processes.
Voting selects representatives and decides issues.
Human rights are fundamental rights belonging to all people.
The Universal Declaration of Human Rights lists basic rights.
Equality means all people have equal rights.
Justice ensures fair treatment under the law.
Peace is the absence of conflict.
Diplomacy resolves disputes through negotiation.
International organizations promote cooperation.
The United Nations works for peace and development.
Culture includes the beliefs, customs, and arts of a society.
Traditions are practices passed down through generations.
Customs are accepted ways of behaving.
Values are principles considered important.
Ethics studies right and wrong behavior.
Morality concerns personal standards of right and wrong.
Law is a system of rules enforced by governments.
Governments create and enforce laws.
Courts interpret and apply laws.
Rights are entitlements protected by law.
Responsibilities are duties expected of citizens.
Communication shares information between people.
Language enables complex communication.
Writing records language in permanent form.
Reading interprets written language.
Speaking produces spoken language.
Listening comprehends spoken language.
Nonverbal communication uses gestures and expressions.
Technology changes how we communicate.
Globalization connects people worldwide.
Diversity includes differences in culture, identity, and experience.
Inclusion ensures all people can participate.
Empathy understands others' feelings and perspectives.
Cooperation achieves goals through working together.
Innovation creates new solutions and ideas.
Creativity produces original and valuable ideas.
Problem-solving finds solutions to challenges.
Learning is the acquisition of knowledge and skills.
Memory stores and retrieves information.
Intelligence enables learning and reasoning.
Wisdom applies knowledge with good judgment.
Happiness is a state of well-being and contentment.
Success is the achievement of goals.
Failure is an opportunity for learning.
Perseverance continues effort despite obstacles.
Resilience recovers from difficulties.
Growth develops abilities over time.
Change is constant in life and society.
Adaptation adjusts to changing conditions.
Balance maintains stability among competing demands.
Purpose gives meaning and direction to life.

"""

# ============================================================
#                    TOKENIZER
# ============================================================

class CharTokenizer:
    """Character-level tokenizer - works offline without downloads"""

    def __init__(self, text):
        # Get unique characters
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for i, ch in enumerate(chars)}

        # Special tokens
        self.pad_token = '\x00'
        self.unk_token = '\x01'

    def encode(self, text):
        return [self.char2idx.get(ch, 0) for ch in text]

    def decode(self, tokens):
        return ''.join([self.idx2char.get(t, '') for t in tokens])

    def __len__(self):
        return self.vocab_size

# ============================================================
#                    MODEL COMPONENTS
# ============================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_seq_len=4096, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cache', emb.cos(), persistent=False)
        self.register_buffer('sin_cache', emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-Head Self-Attention with RoPE"""
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rotary = RotaryEmbedding(self.head_dim, block_size * 2)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), 1).bool())

        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.flash:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=0.1 if self.training else 0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.mask[:T, :T], float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class MLP(nn.Module):
    """Feed-Forward Network with SwiGLU"""
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        hidden = int(n_embd * 4 * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64

        self.gate = nn.Linear(n_embd, hidden, bias=False)
        self.up = nn.Linear(n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """Transformer Block"""
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = Attention(n_embd, n_head, block_size, dropout)
        self.ln2 = RMSNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================
#                    NEURAFORM MODEL
# ============================================================

class Neuraform(nn.Module):
    """
    🧠 NEURAFORM - Large Language Model
    """
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=8, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = RMSNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        # Initialize
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"🧠 Neuraform initialized with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        x = self.tok_emb(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
        self.eval()
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                indices_to_remove = mask.scatter(1, sorted_idx, mask)
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ============================================================
#                    TRAINING SETUP
# ============================================================

print("\n📊 INITIALIZING NEURAFORM")
print("="*60)

# Tokenizer
tokenizer = CharTokenizer(KNOWLEDGE_BASE)
print(f"📚 Vocabulary size: {tokenizer.vocab_size} characters")
print(f"📖 Training data: {len(KNOWLEDGE_BASE):,} characters")

# Encode data
data = torch.tensor(tokenizer.encode(KNOWLEDGE_BASE), dtype=torch.long)
print(f"🔢 Total tokens: {len(data):,}")

# Train/val split
n = int(0.95 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"📈 Train tokens: {len(train_data):,}")
print(f"📉 Val tokens: {len(val_data):,}")

# Model config
BLOCK_SIZE = 256
BATCH_SIZE = 32
N_EMBD = 256
N_HEAD = 8
N_LAYER = 8
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 2000
EVAL_INTERVAL = 200

print(f"\n🔧 Model Configuration:")
print(f"   Block size: {BLOCK_SIZE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Embedding dim: {N_EMBD}")
print(f"   Attention heads: {N_HEAD}")
print(f"   Layers: {N_LAYER}")
print("="*60)

# Create model
model = Neuraform(
    vocab_size=tokenizer.vocab_size,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER,
    block_size=BLOCK_SIZE,
    dropout=DROPOUT
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.1)

# Mixed precision
scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))


# ============================================================
#                    DATA LOADING
# ============================================================

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data_split[i:i+BLOCK_SIZE] for i in ix])
    y = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate():
    model.eval()
    losses = []
    for _ in range(20):
        x, y = get_batch('val')
        with torch.cuda.amp.autocast(enabled=(device=='cuda')):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ============================================================
#                    TRAINING LOOP
# ============================================================

print("\n🚀 TRAINING NEURAFORM")
print("="*60)

start_time = time.time()
best_val_loss = float('inf')

for step in range(MAX_ITERS):
    # Get batch
    x, y = get_batch('train')

    # Forward
    with torch.cuda.amp.autocast(enabled=(device=='cuda')):
        _, loss = model(x, y)

    # Backward
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    # Evaluation
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        val_loss = evaluate()
        elapsed = time.time() - start_time

        status = "✅" if val_loss < best_val_loss else "📊"
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        print(f"{status} Step {step:5d} | Train: {loss.item():.4f} | Val: {val_loss:.4f} | Time: {elapsed:.1f}s")

        # Sample generation
        if step > 0 and step % 400 == 0:
            prompt = "The universe"
            tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
            output = model.generate(tokens, max_tokens=80, temperature=0.8)
            text = tokenizer.decode(output[0].tolist())
            print(f"   💬 Sample: \"{text[:120]}...\"")

total_time = time.time() - start_time
print("="*60)
print(f"✅ TRAINING COMPLETE!")
print(f"⏱️  Total time: {total_time/60:.1f} minutes")
print(f"🏆 Best val loss: {best_val_loss:.4f}")
print("="*60)


# ============================================================
#                    TEXT GENERATION
# ============================================================

print("\n🎭 NEURAFORM TEXT GENERATION")
print("="*60)

def generate_text(prompt, max_tokens=200, temperature=0.8, top_k=50, top_p=0.9):
    tokens = torch.tensor([tokenizer.encode(prompt)], device=device)
    output = model.generate(tokens, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(output[0].tolist())
