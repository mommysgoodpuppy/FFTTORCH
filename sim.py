from concurrent.futures import ThreadPoolExecutor
import logging

# Setup logging for detailed output
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger = logging.getLogger()

class ParallelError(Exception):
    """Custom error for parallel execution violations."""
    pass

class Layer:
    def __init__(self, name, operation):
        self.name = name
        self.operation = operation
        self.input_buffer = None  # Independent buffer for managing input between timesteps
        
    def process(self):
        if self.input_buffer is None:
            raise ParallelError(f"Layer {self.name} has no input to process.")
        
        # Perform the layer operation
        result = self.operation(self.input_buffer)
        logger.debug(f"{self.name} processing input {self.input_buffer} -> {result}")
        return result

class TimeShiftSimulator:
    def __init__(self):
        self.layers = [
            Layer("L1 (+1)", lambda x: x + 1), 
            Layer("L2 (*2)", lambda x: x * 2), 
            Layer("L3 (^2)", lambda x: x ** 2)
        ]
        self.input_journeys = {}
        
    def compute_sequential(self, x):
        """Compute and store true sequential result."""
        result = x
        journey = [x]
        logger.debug(f"\nSequential computation for input {x}:")
        
        for layer in self.layers:
            layer.input_buffer = result  # Set the input buffer
            result = layer.process()
            journey.append(result)
        
        if x not in self.input_journeys:
            self.input_journeys[x] = {}
        self.input_journeys[x]["sequential"] = journey
            
        return result

    def bootstrap_forward(self, x, timestep):
        """Initial bootstrap phase to establish states."""
        logger.debug(f"\nBootstrap forward with input {x} at timestep {timestep}:")
        self.compute_sequential(x)  # Ensure we have sequential results for comparison

        out = x
        parallel_journey = [x]
        
        for i, layer in enumerate(self.layers):
            layer.input_buffer = out  # Bootstrap each layer's input buffer
            out = layer.process()  # Process the initial input
            parallel_journey.append(out)
            logger.debug(f"{layer.name} initial state: input={layer.input_buffer} -> output={out}")

        self.input_journeys[x]["parallel"] = parallel_journey
        self.input_journeys[x]["timestep"] = timestep
        return out

    def parallel_forward(self, new_input, timestep):
        """Parallel forward pass with multithreading and independent layer buffers."""
        logger.debug(f"\nParallel forward with input {new_input} at timestep {timestep}")
        self.compute_sequential(new_input)  # Ensure comparison is available

        # Set up each layer’s input independently from the previous timestep
        self.layers[0].input_buffer = new_input
        parallel_journey = [new_input]
        
        # Execute each layer in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.layers)) as executor:
            futures = {executor.submit(layer.process): i for i, layer in enumerate(self.layers)}
            results = [None] * len(self.layers)

            for future in futures:
                i = futures[future]
                result = future.result()
                results[i] = result
                logger.debug(f"Layer {self.layers[i].name} completed with result {result}")
                # Set the next layer's input if applicable
                if i < len(self.layers) - 1:
                    self.layers[i + 1].input_buffer = result
            
            parallel_journey.extend(results)

        self.input_journeys[new_input]["parallel"] = parallel_journey
        self.input_journeys[new_input]["timestep"] = timestep
        return parallel_journey[-1]
    
    def verify_step(self, timestep):
        """Verify a single timestep against sequential computation."""
        logger.debug(f"\nVerifying timestep {timestep}")
        input_val = timestep
        if input_val not in self.input_journeys:
            logger.debug(f"No data for timestep {timestep}")
            return False
            
        journey = self.input_journeys[input_val]
        seq = journey["sequential"]
        par = journey["parallel"]
        
        logger.debug(f"Input {input_val}:")
        logger.debug(f"  Sequential: {' -> '.join(map(str, seq))}")
        logger.debug(f"  Parallel:   {' -> '.join(map(str, par))}")
        matches = seq == par
        logger.debug(f"  Match: {'PASS' if matches else 'FAIL'}")
        return matches
        
    def verify_all_steps(self):
        logger.debug("\n=== VERIFICATION OF ALL COMPUTATIONS ===")
        all_verified = True
        for timestep in sorted(self.input_journeys.keys()):
            matches = self.verify_step(timestep)
            all_verified = all_verified and matches
            
        logger.debug(f"\nOverall verification: {'PASSED' if all_verified else 'FAILED'}")
        return all_verified

def run_simulation():
    sim = TimeShiftSimulator()
    num_epochs = 10
    timestep = 0

    logger.debug("=== BOOTSTRAP PHASE (3 epochs) ===")
    for epoch in range(3):
        input_val = epoch
        sim.bootstrap_forward(input_val, timestep)
        timestep += 1
            
    logger.debug("\n=== PARALLEL PHASE (10 epochs) ===")
    for epoch in range(num_epochs):
        input_val = epoch + 3
        sim.parallel_forward(input_val, timestep)
        timestep += 1
    
    logger.debug("\n=== PIPELINE CLEARING PHASE ===")
    for epoch in range(3):
        input_val = epoch + num_epochs + 3
        sim.parallel_forward(input_val, timestep)
        timestep += 1
        
    return sim.verify_all_steps()

# Run the simulation and print verification results
run_simulation()
