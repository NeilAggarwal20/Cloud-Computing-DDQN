package research;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.datacenters.DatacenterSimple;
import org.cloudbus.cloudsim.hosts.*;
import org.cloudbus.cloudsim.resources.*;
import org.cloudbus.cloudsim.vms.*;
import org.cloudbus.cloudsim.cloudlets.*;
import org.cloudbus.cloudsim.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudbus.cloudsim.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudbus.cloudsim.schedulers.vm.VmSchedulerTimeShared;
import org.cloudbus.cloudsim.brokers.DatacenterBrokerSimple;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelFull;
import org.cloudbus.cloudsim.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.util.Log;
import ch.qos.logback.classic.Level;
import java.util.*;
import java.nio.file.*;
import java.net.URI;
import java.net.http.*;
import java.time.Duration;

public class Simulation {
    static final int N=200, HOSTS=50, CORES=8;
    static final double PIDLE=100, PMAX=200, HOST_MIPS=2500;
    static final ObjectMapper JSON=new ObjectMapper();
    static final HttpClient HTTP=HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(120)).build();
    static List<Vm> vms=new ArrayList<>();
    static double[] until=new double[N];
    static int[] queued=new int[N];
    static double[] deadline;
    static double[] taskCpu, taskRam, taskBw, taskPriority;
    static int completed=0, violations=0, rr=0;
    static double[] previous=null;
    static int previousAction=-1;
    static double previousPotential=0;
    static String mode;
    static CloudSim sim;
    static double makeBase, energyBase;
    static double gamma=.95;

    @SuppressWarnings("unchecked")
    static Map<String,Object> request(String path, Map<String,Object> data) {
        try {
            var req=HttpRequest.newBuilder(URI.create("http://127.0.0.1:8765/"+path))
                .timeout(Duration.ofMinutes(10)).header("Content-Type","application/json")
                .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(data))).build();
            var response=HTTP.send(req,HttpResponse.BodyHandlers.ofString());
            if(response.statusCode()!=200) throw new IllegalStateException(response.body());
            return JSON.readValue(response.body(),Map.class);
        } catch(Exception e) {throw new RuntimeException("Python bridge failed: "+path,e);}
    }
    static double[] state(double taskLength, double taskDeadline, double cpuUtil,
                          double ramUtil, double bwUtil, double priority) {
        double[] s=new double[N*9]; double now=sim.clock();
        for(int j=0;j<N;j++) {
            double projectedWait=Math.max(0,until[j]-now);
            s[j*9]=Math.min(Math.log1p(projectedWait)/Math.log1p(20000),1);
            s[j*9+1]=Math.min(queued[j]/20.0,1);
            // Include heterogeneous VM speed so the policy can distinguish actions.
            s[j*9+2]=vms.get(j).getMips()/2000.0;
            s[j*9+3]=taskLength/20000.0;
            double finish=Math.max(now,until[j])+taskLength/(vms.get(j).getMips()*cpuUtil);
            double slack=(taskDeadline-finish)/20.0;
            s[j*9+4]=Math.max(-1,Math.min(1,slack));
            s[j*9+5]=cpuUtil; s[j*9+6]=ramUtil; s[j*9+7]=bwUtil;
            s[j*9+8]=priority/3.0;
        }
        return s;
    }
    static double potential(double[] state) {
        double largestProjectedLoad=0;
        for(int j=0;j<N;j++) largestProjectedLoad=Math.max(largestProjectedLoad,state[j*9]);
        return -largestProjectedLoad;
    }
    static Vm choose(Cloudlet c) {
        if(c.isBoundToVm()) return c.getVm();
        int action;
        if(mode.equals("drl") || mode.equals("dqn") || mode.equals("ddqn")) {
            int task=(int)c.getId();
            double[] current=state(c.getLength(),deadline[task],taskCpu[task],taskRam[task],
                taskBw[task],taskPriority[task]);
            double currentPotential=potential(current);
            if(previous!=null) request("observe",Map.of("state",previous,"action",previousAction,
                "reward",gamma*currentPotential-previousPotential,"next_state",current,"done",false));
            action=((Number)request("act",Map.of("state",current)).get("action")).intValue();
            previous=current; previousAction=action; previousPotential=currentPotential;
        } else if(mode.equals("rr")) action=rr++%N;
        else if(mode.equals("eft")) {
            action=0; double best=Double.POSITIVE_INFINITY;
            for(int j=0;j<N;j++) {
            double finish=Math.max(sim.clock(),until[j])+c.getLength()/(vms.get(j).getMips()*taskCpu[(int)c.getId()]);
                if(finish<best) {best=finish; action=j;}
            }
        } else throw new IllegalArgumentException("Mode must be rr, eft, dqn, or ddqn");
        if(action<0||action>=N) throw new IllegalArgumentException("Invalid VM index");
        Vm vm=vms.get(action);
        until[action]=Math.max(sim.clock(),until[action])+c.getLength()/(vm.getMips()*taskCpu[(int)c.getId()]);
        queued[action]++;
        return vm;
    }
    public static void main(String[] args) throws Exception {
        // Contention warnings are emitted for every Cloudlet event and can create
        // enormous logs in overloaded policies; preserve errors without that flood.
        Log.setLevel(Level.ERROR);
        // mode tasks seed output makespanBaseline energyBaseline arrivalRate gamma profilePath
        mode=args.length>0?args[0]:"rr";
        int count=args.length>1?Integer.parseInt(args[1]):100;
        long seed=args.length>2?Long.parseLong(args[2]):42;
        Path out=Path.of(args.length>3?args[3]:"results/rr.json");
        makeBase=args.length>4?Double.parseDouble(args[4]):100;
        energyBase=args.length>5?Double.parseDouble(args[5]):1;
        double rate=args.length>6?Double.parseDouble(args[6]):20;
        gamma=args.length>7?Double.parseDouble(args[7]):.95;
        Path profile=Path.of(args.length>8?args[8]:"data/workload-profile.csv");
        if(count<1||rate<=0||makeBase<=0||energyBase<=0||gamma<=0||gamma>1) throw new IllegalArgumentException("Positive parameters required; gamma must be in (0,1]");
        sim=new CloudSim();
        for(int dc=0;dc<2;dc++) {
            List<Host> hosts=new ArrayList<>();
            for(int h=0;h<25;h++) {
                List<Pe> pes=new ArrayList<>();
                for(int p=0;p<CORES;p++) pes.add(new PeSimple(HOST_MIPS));
                Host host=new HostSimple(65536,100000,1000000,pes);
                host.setVmScheduler(new VmSchedulerTimeShared()); hosts.add(host);
            }
            new DatacenterSimple(sim,hosts).setSchedulingInterval(1.0);
        }
        var broker=new DatacenterBrokerSimple(sim);
        broker.setVmDestructionDelay(1000.0);
        for(int i=0;i<N;i++) {
            Vm vm=new VmSimple(i,1000+(i%3)*500,1);
            // Match the documented 1,000-unit VM bandwidth demand. Requesting
            // all host bandwidth here limited the datacenter to one VM per host.
            vm.setRam(16384).setBw(1000).setSize(100000);
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared()); vms.add(vm);
        }
        broker.setVmMapper(Simulation::choose);
        broker.submitVmList(vms);
        System.out.println("VMs submitted: " + vms.size());
        Random random=new Random(seed); deadline=new double[count];
        taskCpu=new double[count]; taskRam=new double[count]; taskBw=new double[count]; taskPriority=new double[count];
        List<String[]> profileRows=Files.readAllLines(profile).stream().skip(1).map(line->line.split(",")).toList();
        if(profileRows.size()!=count) throw new IllegalArgumentException("Workload profile rows="+profileRows.size()+", expected="+count+": "+profile);
        List<Cloudlet> tasks=new ArrayList<>(); double arrival=0;
        for(int i=0;i<count;i++) {
            if(i>0) arrival+=-Math.log(1-random.nextDouble())/rate;
            String[] row=profileRows.get(i);
            long length=Long.parseLong(row[0]);
            taskCpu[i]=Double.parseDouble(row[1]); taskRam[i]=Double.parseDouble(row[2]);
            taskBw[i]=Double.parseDouble(row[3]); taskPriority[i]=Double.parseDouble(row[4]);
            double executionSeconds=Double.parseDouble(row[5]);
            Cloudlet c=new CloudletSimple(i,length,1);
            c.setFileSize(1)
 .setOutputSize(1)
 .setUtilizationModelCpu(new UtilizationModelDynamic(taskCpu[i]))
 .setUtilizationModelRam(new UtilizationModelDynamic(taskRam[i]))
 .setUtilizationModelBw(new UtilizationModelDynamic(taskBw[i]));
            c.setSubmissionDelay(arrival);
            deadline[i]=arrival+1.5*executionSeconds+2;
            final int id=i;
            c.addOnFinishListener(info->{
                completed++; if(info.getCloudlet().getFinishTime()>deadline[id]) violations++;
                queued[(int)info.getCloudlet().getVm().getId()]--;
            });
            tasks.add(c);
        }
        broker.submitCloudletList(tasks); sim.start();
        List<Cloudlet> finished = broker.getCloudletFinishedList();

if (finished.size() != count) {
    Set<Integer> done = new HashSet<>();
    for (Cloudlet c : finished) done.add((int) c.getId());

    for (Cloudlet task : tasks) {
        if (!done.contains((int) task.getId())) {
            System.out.printf(
                "UNFINISHED task=%d status=%s vm=%s startTime=%s finishTime=%s%n",
                task.getId(),
                task.getStatus(),
                task.getVm() == null ? -1 : task.getVm().getId(),
                task.getExecStartTime(),
                task.getFinishTime()
            );
        }
    }

    throw new IllegalStateException(
        "Not all tasks completed: finished=" + finished.size() + ", expected=" + count
    );
}
        double makespan=finished.stream().mapToDouble(Cloudlet::getFinishTime).max().orElseThrow();
        // Integrate the linear host-power model, all hosts always powered on.
        double joules=HOSTS*PIDLE*makespan;
        for(Cloudlet c:finished) joules+=(PMAX-PIDLE)*c.getActualCpuTime()*taskCpu[(int)c.getId()]*c.getVm().getMips()/(HOST_MIPS*CORES);
        double energy=joules/3600000.0, sla=violations/(double)count, throughput=count/makespan;
        // RR defines the common baseline. Score RR itself against its own metrics.
        double rewardMakespanBase=mode.equals("rr")?makespan:makeBase;
        double rewardEnergyBase=mode.equals("rr")?energy:energyBase;
        double throughputBase=count/rewardMakespanBase;
        // Bounded relative scores keep an unexpectedly poor policy from creating huge targets.
        double makespanScore=(rewardMakespanBase-makespan)/(rewardMakespanBase+makespan);
        double energyScore=(rewardEnergyBase-energy)/(rewardEnergyBase+energy);
        double throughputScore=(throughput-throughputBase)/(throughput+throughputBase);
        double reward=.35*makespanScore+.30*(1-sla)+.20*energyScore+.15*throughputScore;
        if((mode.equals("drl") || mode.equals("dqn") || mode.equals("ddqn")) && previous!=null) request("observe",Map.of("state",previous,
            "action",previousAction,"reward",reward-previousPotential,
            "next_state",state(0,sim.clock(),1,0,0,1),"done",true));
        Map<String,Object> result=new LinkedHashMap<>();
        result.put("mode",mode);result.put("tasks",count);result.put("seed",seed);
        result.put("makespan_s",makespan);result.put("energy_kwh",energy);
        result.put("sla_violation_pct",100*sla);result.put("throughput_tasks_s",throughput);
        result.put("reward",reward);result.put("submitted",count);result.put("completed",completed);
        result.put("failed",count-completed);result.put("completion_rate_pct",100.0*completed/count);
        if(out.getParent()!=null) Files.createDirectories(out.getParent());
        JSON.writerWithDefaultPrettyPrinter().writeValue(out.toFile(),result);
        List<String> rows=new ArrayList<>(List.of("task_id,vm_id,length_mi,arrival_s,start_s,finish_s,deadline_s"));
        for(Cloudlet c:finished) rows.add(c.getId()+","+c.getVm().getId()+","+c.getLength()+","+
            c.getSubmissionDelay()+","+c.getExecStartTime()+","+c.getFinishTime()+","+deadline[(int)c.getId()]);
        Files.write(Path.of(out.toString().replace(".json","-tasks.csv")),rows);
    }
}
