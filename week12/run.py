#!/usr/bin/env python3
"""CloudShell Bedrock labs: Python standard library orchestrates the installed AWS CLI.
No boto3, pip, access keys, or local installation. Default mode replays verified CloudShell CLI results.
"""
import argparse, concurrent.futures, csv, datetime, html, json, os, random, re
import statistics, subprocess, tempfile, time, uuid, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT/'workshop_data.json').read_text())
LITE = 'amazon.nova-lite-v1:0'
MICRO = 'us.amazon.nova-micro-v1:0'
REGION = 'us-west-2'

def readlines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def writejson(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def payload(prompt, cap=96):
    return {'messages':[{'role':'user','content':[{'text':prompt}]}],
            'inferenceConfig':{'maxTokens':cap,'temperature':0,'topP':0.9}}

def response_text(response):
    return ''.join(c.get('text','') for c in (response or {}).get('output',{}).get('message',{}).get('content',[]))

def invoke(model, body):
    started=time.perf_counter(); errors=[]; response=None
    environment=dict(os.environ, AWS_PAGER='', AWS_MAX_ATTEMPTS='1', AWS_RETRY_MODE='standard')
    with tempfile.TemporaryDirectory(prefix='bedrock-request-') as folder:
        request=Path(folder)/'request.json'; writejson(request,dict(modelId=model,**body))
        for attempt in range(1,4):
            try:
                run=subprocess.run(['aws','bedrock-runtime','converse','--region',REGION,
                    '--cli-input-json','file://'+str(request),'--output','json','--no-cli-pager',
                    '--cli-connect-timeout','10','--cli-read-timeout','90'],
                    capture_output=True,text=True,env=environment,timeout=105)
                if run.returncode==0:
                    response=json.loads(run.stdout); break
                error=run.stderr.strip()
            except subprocess.TimeoutExpired:
                error='Client timeout; request outcome and usage unknown'
            errors.append(error)
            transient=any(code in error for code in ['ThrottlingException','ServiceUnavailableException','InternalServerException','ModelNotReadyException'])
            if not transient or attempt==3: break
            time.sleep(min(4,2**(attempt-1))+random.uniform(0,0.3))
    return dict(model=model,response=response,latency_s=time.perf_counter()-started,
                attempts=attempt,errors=errors,source='live AWS CLI')

def pairs_hook(items):
    result={}
    for k,v in items:
        if k in result: raise ValueError('duplicate key')
        result[k]=v
    return result

def score(text, expected):
    try:
        obj=json.loads(text,object_pairs_hook=pairs_hook)
        valid=isinstance(obj,dict) and set(obj)=={'research_area','method','dataset'}
        valid=valid and all(isinstance(obj[k],str) and obj[k].strip() for k in ('research_area','method'))
        valid=valid and (obj['dataset'] is None or isinstance(obj['dataset'],str) and bool(obj['dataset'].strip()))
        norm=lambda v: ' '.join(v.lower().split()) if isinstance(v,str) else v
        exact=valid and all(norm(obj[k])==norm(expected[k]) for k in ('research_area','method','dataset'))
        return bool(valid),bool(exact)
    except (ValueError,TypeError,KeyError): return False,False

def row(call, expected=None):
    response=call.get('response') or {}; usage=response.get('usage',{})
    text=response_text(response); valid,exact=score(text,expected) if expected else (None,None)
    rates=DATA['rates']['models'].get(call['model'],{})
    estimated=sum(usage.get(field,0)*rates.get(rate,0) for field,rate in
        [('inputTokens','input'),('outputTokens','output'),('cacheReadInputTokens','cache_read'),('cacheWriteInputTokens','cache_write')])/1e6 if usage else None
    return dict(model=call['model'],id=call.get('abstract_id',call.get('question_id','')),
        success=bool(call.get('response')),attempts=call.get('attempts',1),
        retries=max(0,call.get('attempts',1)-1),client_s=round(call['latency_s'],4),
        service_ms=response.get('metrics',{}).get('latencyMs'),
        input_tokens=usage.get('inputTokens'),output_tokens=usage.get('outputTokens'),
        cache_write=usage.get('cacheWriteInputTokens'),cache_read=usage.get('cacheReadInputTokens'),
        stop=response.get('stopReason',''),valid_json=valid,exact=exact,
        estimated_usd=estimated,text=text,error=' | '.join(call.get('errors',[])))

def percentile(values, fraction):
    if not values: return None
    values=sorted(values); at=(len(values)-1)*fraction; low=int(at); high=min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(at-low)

def summarise(rows, elapsed):
    okay=[r for r in rows if r['success']]; latencies=[r['client_s'] for r in okay]
    estimated=sum(r['estimated_usd'] or 0 for r in rows); exact=sum(r['exact'] is True for r in rows)
    return dict(submitted=len(rows),successes=len(okay),failures=len(rows)-len(okay),
        retries=sum(r['retries'] for r in rows),wall_s=round(elapsed,4),
        success_per_s=round(len(okay)/elapsed,4) if elapsed else None,
        p50_client_s=round(statistics.median(latencies),4) if latencies else None,
        p95_client_s=round(percentile(latencies,.95),4) if latencies else None,
        valid_json=sum(r['valid_json'] is True for r in rows),exact=exact,
        estimated_usd=round(estimated,8),costed_responses=sum(r['estimated_usd'] is not None for r in rows),
        estimated_usd_per_exact=round(estimated/exact,8) if exact else None)

def table(rows, fields=None):
    if not rows: return
    fields=fields or list(rows[0]); strings=[[str(r.get(f,'-') if r.get(f) is not None else '-') for f in fields] for r in rows]
    widths=[max(len(f),*(len(r[i]) for r in strings)) for i,f in enumerate(fields)]
    print('  '.join(f.ljust(w) for f,w in zip(fields,widths)))
    for r in strings: print('  '.join(v.ljust(w) for v,w in zip(r,widths)))

def save_csv(path, rows):
    if not rows: return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def html_table(rows, fields=None):
    if not rows: return '<p>No rows.</p>'
    fields=fields or list(rows[0]); esc=lambda s:html.escape(str(s if s is not None else '-'))
    return '<table><thead><tr>'+''.join('<th>'+esc(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(r.get(k))+'</td>' for k in fields)+'</tr>' for r in rows)+'</tbody></table>'

def finish(folder, title, source, summaries, rows, raw):
    save_csv(folder/'summary.csv',summaries); save_csv(folder/'observations.csv',rows)
    (folder/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
    chart=''
    if summaries and 'workers' in summaries[0]:
        chart='<h2>Successful requests per second</h2>'
        maximum=max((r['success_per_s'] or 0) for r in summaries) or 1
        for r in summaries:
            chart+=f'<div>{r["workers"]} workers &nbsp; <b>{r["success_per_s"]}</b><div style="height:18px;background:#167e85;width:{90*(r["success_per_s"] or 0)/maximum}%"></div></div>'
    detail_fields=['id','success','client_s','service_ms','valid_json','exact','text','error']
    if title=='cache': detail_fields=['arm','id','success','client_s','cache_write','cache_read','answer_correct','text','error']
    page='<!doctype html><meta charset="utf-8"><title>Bedrock '+title+' results</title><style>body{font:16px/1.5 Arial;margin:40px;color:#18353a}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #cad6d7;padding:9px;text-align:left;vertical-align:top}th{background:#e6f2f1}h1,h2{color:#126a70}p{max-width:1000px}td{max-width:460px;overflow-wrap:anywhere}</style>'
    page+='<h1>Bedrock '+title+' results</h1><p><b>Source: '+html.escape(source)+'</b></p>'
    page+='<p>Client latency includes client work, networking, all attempts and retry delays. Live CLI runs include a new CLI process for each attempt. Saved examples use the same AWS CLI workflow. Keep separate runs and their source labels distinct. AWS service_ms is separate.</p>'
    page+='<p>Estimated USD uses the included 2026-11-19 price snapshot and returned usage only. Failed or timed-out attempts may have unreported charges. It excludes warm-up, S3, taxes and unrelated usage. Review current rates before relying on these estimates. A missing cache counter is not proof that implicit caching was absent.</p>'
    page+=html_table(summaries)+chart+'<h2>Individual observations</h2>'+html_table(rows,detail_fields)
    (folder/'report.html').write_text(page)
    archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in folder.iterdir(): z.write(path,path.name)
    print('\nResults folder:',folder)
    print('Download this ZIP:',archive)
    print('Extract it on your computer and open report.html. No notebook is required.')

def prepare():
    directory=ROOT/'requests'; directory.mkdir(exist_ok=True)
    abstracts=DATA['abstracts']; prompt=DATA['extraction_prompt']
    writejson(directory/'quickstart.json',dict(modelId=LITE,**payload('In one sentence, explain what a GPU is used for in deep learning.',32)))
    (directory/'playground-A01.txt').write_text(prompt+abstracts[0]['abstract']+'\n')
    # Bedrock batch jobs need at least 100 records, so the 12 abstracts repeat.
    batch=[dict(recordId=f"{abstracts[i%len(abstracts)]['id']}-{i+1:04d}",modelInput=payload(prompt+abstracts[i%len(abstracts)]['abstract']))
           for i in range(100)]
    (directory/'batch_input.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in batch))
    print('Ready: requests/quickstart.json, requests/playground-A01.txt, requests/batch_input.jsonl')

def run(args):
    if args.lab=='prepare': prepare(); return
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,24}',args.group): raise SystemExit('Use a label such as test-01 (lowercase letters, numbers, hyphens)')
    if not 8<=args.requests<=120: raise SystemExit('--requests must be 8 through 120')
    key={x['id']:x for x in DATA['answer_key']}
    abstracts=DATA['abstracts']; prompt=DATA['extraction_prompt']
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:4]
    folder=ROOT/'results'/f'{args.group}-{args.lab}-{stamp}'; folder.mkdir(parents=True)
    source='LIVE AWS CLI, Oregon' if args.live else 'SAVED AWS CLI run, 2026-11-19'
    print('Source:',source,flush=True); rows=[]; summaries=[]; raw=[]
    metadata=dict(source=source,region=REGION,lab=args.lab,group=args.group,
        requests_per_setting=args.requests if args.live and args.lab=='sweep' else 24 if args.lab=='sweep' else None,
        workers=[1,2,4,8] if args.lab=='sweep' else None,
        max_output_tokens=24 if args.lab=='cache' else 96,temperature=0,top_p=.9,
        automatic_cli_attempts=1,maximum_application_attempts=3)
    if args.live:
        metadata['aws_cli_version']=subprocess.run(['aws','--version'],capture_output=True,text=True).stdout.strip()
    writejson(folder/'run_metadata.json',metadata)
    if args.lab=='sweep':
        if args.live:
            warm=invoke(LITE,payload('Reply READY.',8)); writejson(folder/'warmup.json',warm)
            if warm['response'] is None: raise SystemExit('Warm-up failed. '+str(warm['errors']))
            runs=[]
            for workers in (1,2,4,8):
                jobs=[abstracts[i%len(abstracts)] for i in range(args.requests)]
                bodies=[payload(prompt+item['abstract']) for item in jobs]
                start=time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    calls=list(pool.map(lambda body:invoke(LITE,body),bodies))
                elapsed=time.perf_counter()-start
                for item,call in zip(jobs,calls): call['abstract_id']=item['id']
                runs.append(dict(workers=workers,elapsed_s=elapsed,calls=calls))
                (folder/'checkpoint.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in runs))
                print(f'{workers} workers finished in {elapsed:.2f} seconds',flush=True)
        else: runs=DATA['saved']['sweep']
        raw=runs
        for run_data in runs:
            group=[dict(workers=run_data['workers'],**row(c,key[c['abstract_id']])) for c in run_data['calls']]
            rows.extend(group); summaries.append(dict(workers=run_data['workers'],**summarise(group,run_data['elapsed_s'])))
        table(summaries,['workers','successes','failures','retries','success_per_s','p95_client_s','exact'])
    elif args.lab=='selection':
        if args.live:
            for model in (LITE,MICRO):
                for item in abstracts:
                    call=invoke(model,payload(prompt+item['abstract'])); call['abstract_id']=item['id']; raw.append(call)
        else: raw=DATA['saved']['selection']
        rows=[row(c,key[c['abstract_id']]) for c in raw]
        for model in (LITE,MICRO):
            group=[r for r in rows if r['model']==model]
            summaries.append(dict(model=model,**summarise(group,sum(r['client_s'] for r in group))))
        table(summaries,['model','submitted','successes','valid_json','exact'])
    elif args.lab=='cache':
        questions=DATA['cache_questions']
        if args.live:
            document=DATA['cache_manual']
            prefix='Run '+uuid.uuid4().hex+'\nUse this reference to answer each question.\n'+document
            for arm in ('baseline','checkpoint'):
                for q in questions:
                    body=payload(q['question'],24); body['system']=[{'text':prefix}]
                    if arm=='checkpoint': body['system'].append({'cachePoint':{'type':'default'}})
                    call=invoke(LITE,body); call.update(arm=arm,question_id=q['id']); raw.append(call)
                    (folder/'checkpoint.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
        else: raw=DATA['saved']['cache']
        answers={q['id']:q['answer'] for q in questions}
        rows=[dict(arm=c['arm'],**row(c),answer_correct=response_text(c.get('response')).strip()==answers[c['question_id']]) for c in raw]
        for arm in ('baseline','checkpoint'):
            group=[r for r in rows if r['arm']==arm]
            summaries.append(dict(arm=arm,successes=sum(r['success'] for r in group),correct=sum(r['answer_correct'] for r in group),
                cache_hits=sum((r['cache_read'] or 0)>0 for r in group),reported_write_tokens=sum(r['cache_write'] or 0 for r in group),
                reported_read_tokens=sum(r['cache_read'] or 0 for r in group),median_client_s=round(statistics.median(r['client_s'] for r in group),4),
                estimated_usd=round(sum(r['estimated_usd'] or 0 for r in group),8)))
        table(rows,['arm','id','client_s','cache_write','cache_read','answer_correct']); print(); table(summaries)
    finish(folder,args.lab,source,summaries,rows,raw)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('lab',choices=['prepare','selection','sweep','cache'])
    p.add_argument('--live',action='store_true',help='Send billable requests; default is saved replay')
    p.add_argument('--group',default='test-01'); p.add_argument('--requests',type=int,default=24)
    run(p.parse_args())
