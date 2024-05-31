finite-inf.json: finite.json inf.json
	boss finite.json --compose-cyclic inf.json --eliminate-states >$@

f.json: finite.json
	boss $< >$@

finite-inf.math: f.json finite-inf.json
	./json2matrix.js -m f.json -s 0..2 -p '$$F=' | perl -pe 's/alpha/α/g;s/beta/β/g' >$@
	./json2matrix.js -m finite-inf.json -s 0..14 -p '$$FG=' | perl -pe 's/alpha/α/g;s/beta/β/g' >>$@
	echo 'matStates$$FG = {1}' >>$@
	echo 'ins1States$$FG = {4,6,8,10}' >>$@
	echo 'ins2States$$FG = {5,7,9,11}' >>$@
	echo 'del1States$$FG = {2}' >>$@
	echo 'del2States$$FG = {3}' >>$@
	echo 'nullStates$$FG = {12,13,14,15}' >>$@

finite-inf.tex: finite-inf.json
	./json2matrix.js -m finite-inf.json -s 0..14 -t >$@

%.boss.dot: %.json
	boss $< -G >$@

%.dot: %.boss.dot
	cat $< | perl -pe 's/\*0.50+/\/2/g;s/2.0+/2/g;s/\)\*\(/)(/g;s/\{.*?wait..:(.*?)\}/\1/;s/(\[label=.)...(\w+).....(\w+).../\1\2\3/g;s/alpha/α/g;s/beta/β/g;s/headlabel="&epsilon;.X"/color="green"/;s/headlabel="X.X"/color="orange"/;s/headlabel="X.&epsilon;"/color="red"/;s/headlabel="&epsilon;.&epsilon;",?//;s/\/2\*(.*)"/\1\/2"/;s/\*//g;' >$@

%.pdf: %.dot
	dot -Tpdf $< >$@

%.pdf.open: %.pdf
	open $<

.SECONDARY:
